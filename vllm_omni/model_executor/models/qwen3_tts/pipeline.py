# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS pipeline: Talker (text → RVQ codec) → Code2Wav (codec → audio).

Chunked vs end-to-end mode is dispatched from ``deploy.async_chunk``.
"""
# 模块说明：Qwen3-TTS 的流水线拓扑定义。
# 标准两阶段流水线：Talker（文本 → RVQ 音频编解码 token）→ Code2Wav（编解码 token → 波形音频）。
# 分块（chunked）模式与端到端（end-to-end）模式由部署配置 ``deploy.async_chunk`` 决定走哪条路径。

from vllm_omni.config.stage_config import (
    PipelineConfig,         # 整条流水线的拓扑配置（冻结的 dataclass）
    StageExecutionType,     # 阶段执行类型枚举：LLM_AR（自回归）/ LLM_GENERATION（一次性生成）/ DIFFUSION（扩散）
    StagePipelineConfig,    # 单个阶段的固定拓扑配置
)
from vllm_omni.outputs.output_modality import (
    TensorAccumulationStrategy,             # 张量累积策略枚举：多步输出如何合并（拼接 / 替换等）
    register_key_accumulation_strategy,     # 注册函数：为某个输出 key 指定跨解码步的累积方式
)

# 输入处理函数所在模块的点路径前缀，后面的字符串配置会基于它拼出完整函数路径。
# 框架通过「字符串点路径 + 动态 import」的方式调用这些 hook，避免在配置里直接持有函数对象。
_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"

# The talker's codec-frame keys are not waveform chunks, so they must not go
# through the AUDIO modality's waveform-tuned default (CONCAT_LAST) whenever
# a stage carrying them consolidates its output under that modality (e.g. an
# AUDIO-typed final stage that forwards them directly instead of routing
# through code2wav): ``codes.audio`` grows one row per decode step along
# dim 0, and ``codes.ref`` re-emits the constant reference-context matrix
# unchanged at every step.
# 中文概括：Talker 输出的 codec 帧 key 不是波形块，不能套用 AUDIO 模态为波形设计的默认拼接策略。
#   codes.audio：每个解码步沿 dim 0 增加一行 → 用 CONCAT_DIM0 逐行拼接。
#   codes.ref：每步重复发出同一份不变的参考上下文矩阵 → 用 REPLACE 直接覆盖即可。
register_key_accumulation_strategy("codes.audio", TensorAccumulationStrategy.CONCAT_DIM0)
register_key_accumulation_strategy("codes.ref", TensorAccumulationStrategy.REPLACE)
# code2wav's client key for the frames it decoded: [frames, codebooks] per step.
# code2wav 阶段解码输出的 key：每步产生 [帧数, 码本数] 的张量，跨步沿 dim 0 拼接成完整序列。
register_key_accumulation_strategy("codec_frames", TensorAccumulationStrategy.CONCAT_DIM0)

# ==================== 标准两阶段流水线：Talker → Code2Wav ====================
QWEN3_TTS_PIPELINE = PipelineConfig(
    model_type="qwen3_tts",                          # 模型类型标识：StageConfigFactory 据此把模型路由到本流水线
    default_deploy_config_name="qwen3_tts.yaml",     # 默认部署配置文件（从 vllm_omni/deploy 目录加载，提供并行/显存等默认值）
    # Pipeline-level default; the code2wav stage overrides per-stage below.
    model_arch="Qwen3TTSTalkerForConditionalGeneration", # 所有AR的默认骨架  # 流水线级默认模型架构类；code2wav 阶段在下方用自己的 model_arch 覆盖它
    stages=(                                         # 阶段元组：按拓扑声明两个执行阶段
        # ---------- 阶段 0：Talker，自回归生成 RVQ codec token ----------
        StagePipelineConfig(
            stage_id=0,                              # 阶段编号：0 号即流水线入口
            model_stage="qwen3_tts",                 # 阶段名称标识
            execution_type=StageExecutionType.LLM_AR,# 执行类型：自回归 LLM，逐 token 解码（由 vLLM 引擎服务）
            input_sources=(), # 路由:数据的输入      # 上游来源为空元组 → 本阶段直接接收用户请求，是数据流起点
            owns_tokenizer=True,                     # 本阶段持有 tokenizer：负责对输入文本做 tokenize
            engine_output_type="latent",             # 引擎输出标记为 latent：产出的是中间结果（codec），不是最终交付物
            # async_chunk（分块流式）模式下，把本阶段输出加工为下一阶段输入的 hook：
            # Talker 每解出一块 codec 就即时传给 code2wav，实现边生成边解码的流式流水线。
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            supports_native_mrv2_data_plane=True, # 适配MRv2的数据传输路径  # 声明支持原生 MRv2 数据面（worker 间直传，少绕 orchestrator）
            # 非分块（完整负载）模式下的跨阶段 hook：等 Talker 全部解码完，一次性把完整 codec 负载交给 code2wav。
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            sampling_constraints={                   # 采样约束：生成时强制套用的参数
                "detokenize": False,                 # 不做 detokenize：输出是 codec token 而非文本，无需还原成文字
                "stop_token_ids": [2150],            # 遇到 token id 2150 即停止解码（音频结束符）
            },
            # talker_mtp sums the residual codebook embeddings and adds
            # text_step; neither is in the token stream (vllm-project/vllm-omni#6179).
            # 中文概括：talker_mtp 的解码步还会额外消费「残差码本 embedding 求和 + text_step」这类旁路信息，
            # 它们不在 token 流里，抢占后无法靠重放 prompt 重建 KV，因此禁止重计算式抢占，直接失败报错。
            recompute_preemption="fail",
        ),
        # ---------- 阶段 1：Code2Wav，把 codec token 一次性解码为波形音频 ----------
        StagePipelineConfig(
            stage_id=1,                              # 阶段编号：1
            model_stage="code2wav",                  # 阶段名称标识
            execution_type=StageExecutionType.LLM_GENERATION,  # 执行类型：一次性生成（非自回归，跑完整生成流程出音频）
            input_sources=(0,),                      # 上游来源：阶段 0（Talker）的输出即本阶段输入
            final_output=True,                       # 标记为最终输出阶段：流水线终点，结果返回给用户
            final_output_type="audio",               # 最终输出模态：音频
            engine_output_type="audio",              # 引擎输出同样按 audio 模态打标，供 API 输出处理器识别
            model_arch="Qwen3TTSCode2Wav",           # 覆盖流水线默认架构：本阶段加载 Code2Wav 模型类
            supports_native_mrv2_data_plane=True,    # 同样支持原生 MRv2 数据面传输
            # ``sync_process_input_func`` is the only input-proc override for
            # this stage in sync (non-async-chunk) mode: a length-only
            # ``_token_only`` placeholder.  The bulk codec payload itself
            # ships via the worker connector from stage 0's
            # ``talker2code2wav_full_payload`` producer.  Under async_chunk
            # mode no pre-stage processing is needed -- chunks deliver
            # directly to the consumer.
            # 中文概括：同步模式下本阶段唯一的输入预处理是 talker2code2wav_token_only——只生成一个
            # 「仅长度」的占位输入；真正的 codec 大负载由阶段 0 经 worker connector 直送。
            # async_chunk 模式下无需预处理，数据块直接送达消费者。
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True}, # 本阶段做 detokenize：把生成结果还原为最终音频输出
            extras={"tts_args": {"max_instructions_length": 500}},  # 额外参数：TTS 指令文本的最大长度为 500
            requires_full_payload_input=True,        # 非 async 路径下，必须等上游完整负载经 connector 到齐后才调度本阶段
        ),
    ),
)


# Single-stage variant: the Talker decodes each frame with the stateful
# streaming codec decoder and emits PCM as the final output (enable with the
# ``talker_stream_decode`` model option). No Code2Wav stage, no connector.
# 中文概括：单阶段融合变体——Talker 内部用有状态的流式 codec 解码器逐帧解码，直接输出 PCM 音频。
# 通过模型选项 ``talker_stream_decode`` 启用；不再需要独立的 Code2Wav 阶段，也就没有跨阶段 connector。
# ==================== 单阶段融合流水线：Talker 内部直接出音频 ====================
QWEN3_TTS_FUSED_PIPELINE = PipelineConfig(
    model_type="qwen3_tts_fused",                            # 独立的模型类型标识，与两阶段版本区分开
    default_deploy_config_name="qwen3_tts_fused_single_gpu.yaml",  # 默认部署配置：单 GPU 的融合部署方案
    single_stage_async_chunk=True,                           # 单阶段自身以 async-chunk 方式流式产出最终音频（保留 deploy.async_chunk 行为）
    model_arch="Qwen3TTSTalkerForConditionalGeneration",     # 唯一的阶段沿用 Talker 架构类
    stages=(                                                 # 只有一个阶段
        # ---------- 阶段 0：Talker 融合流式解码，直接产出最终 PCM 音频 ----------
        StagePipelineConfig(
            stage_id=0,                                      # 阶段编号：0，同时也是唯一起点与终点
            model_stage="qwen3_tts",                         # 阶段名称标识
            execution_type=StageExecutionType.LLM_AR,        # 自回归 LLM 解码
            input_sources=(),                                # 无上游：直接接收用户请求
            owns_tokenizer=True,                             # 持有 tokenizer，负责输入文本的 tokenize
            final_output=True,                               # 本阶段即最终输出阶段（融合模式下 Talker 直接出音频）
            final_output_type="audio",                       # 最终输出模态：音频
            # The API output processor tags ``model_outputs`` by this modality.
            # API 输出处理器依据这个模态标记给 ``model_outputs`` 打标签，走音频输出通道。
            engine_output_type="audio",
            supports_running_prefix_cache_reset=False,       # 不支持运行中的前缀缓存重置：有状态的流式音频解码无法回滚已消费的帧
            supports_native_mrv2_data_plane=True,            # 支持原生 MRv2 数据面
            sampling_constraints={                           # 采样约束（与两阶段版本的 Talker 一致）
                "detokenize": False,                         # 不做 detokenize：输出 codec，由内部流式解码器转音频
                "stop_token_ids": [2150],                    # token 2150 为音频结束符
            },
            # talker_mtp sums the residual codebook embeddings and adds
            # text_step; neither is in the token stream (vllm-project/vllm-omni#6179).
            # 中文概括：同两阶段版本——旁路信息不在 token 流中，抢占后无法重放重建 KV，禁止重计算式抢占。
            recompute_preemption="fail",
        ),
    ),
)
