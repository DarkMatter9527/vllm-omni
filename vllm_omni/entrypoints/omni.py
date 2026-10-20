# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

from tqdm.auto import tqdm
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import OutputMessage
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams

logger = init_logger(__name__)


class Omni(OmniBase):
    """Synchronous entrypoint for offline generation."""
    
    # 这里才是引擎的创建
    def _create_engine(self, **engine_kwargs: Any) -> AsyncOmniEngine:
        return AsyncOmniEngine(**engine_kwargs)

    def _maybe_force_final_only_for_llm_stages(
        self,
        sampling_params_list: Sequence[OmniSamplingParams],
    ) -> list[OmniSamplingParams]:
        """Return per-stage params with LLM stages forced to FINAL_ONLY.

        The caller may explicitly request ``output_kind = DELTA`` on a stage to
        opt into streaming; such stages are left alone.  All other LLM stages
        are forced to FINAL_ONLY.
        """
        effective_params: list[OmniSamplingParams] = []
        for stage_id, params in enumerate(sampling_params_list):
            sp = copy.deepcopy(params)
            stage_meta = self.engine.get_stage_metadata(stage_id)
            if (
                stage_meta.stage_type != "diffusion"
                and hasattr(sp, "output_kind")
                and sp.output_kind != RequestOutputKind.DELTA
            ):
                sp.output_kind = RequestOutputKind.FINAL_ONLY
            effective_params.append(sp)
        return effective_params

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[True],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]: ...

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[False] = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[OmniRequestOutput]: ...

    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: bool = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None] | list[OmniRequestOutput]:
        # Expand sampling params for PD disaggregation (user may provide N-1 params)
        """
        omni生成主逻辑:
            第一步:获取采样参数列表
            第二步:omni框架执行并返回
        """
        
        """ 1. 获取采样参数 """
        if (
            sampling_params_list is not None
            and isinstance(sampling_params_list, Sequence)
            and not isinstance(sampling_params_list, (str, bytes))
        ):
            sampling_params_list = self._maybe_expand_sampling_params(list(sampling_params_list))
        sampling_params_list = self.resolve_sampling_params_list(sampling_params_list) # 获取采样参数列表,每一个stage一个参数配置

        """ 2. omni框架执行并返回 """
        try:
            if py_generator:
                return self._run_generation_with_generator(prompts, sampling_params_list, use_tqdm)
            return list(self._run_generation(prompts, sampling_params_list, use_tqdm)) # 返回处理结果

        except Exception as e:
            logger.exception("[Omni] Failed to run generation: %s", e)
            self.close()
            raise

    def _run_generation_with_generator(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        # 核心还是_run_generation
        yield from self._run_generation(
            prompts,
            sampling_params_list,
            use_tqdm,
        )

    def _run_generation(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        try:
            """1. 配置采样参数"""
            sampling_params_list = self._maybe_force_final_only_for_llm_stages(sampling_params_list) # 没有显式要求流式输出的时候，强制所有 LLM 类型的 stage 只输出最终结果，不输出中间过程

            # prompt包装为列表形式，方便直接使用for循环取用
            if isinstance(prompts, str) or not isinstance(prompts, Sequence):
                request_prompts: list[OmniPromptType] = [prompts]
            else:
                request_prompts = list(prompts)
            if not request_prompts:
                return

            request_ids = [f"{i}_{uuid.uuid4()}" for i in range(len(request_prompts))] # 给每一个请求一个ID值
            req_start_ts: dict[str, float] = {} # 时间记录
            wall_start_ts = time.time()
            req_final_stage_ids: dict[str, int] = {} # 映射: 请求ID -->> 改请求的输出从哪些stage中取得

            """2. 将请求添加到引擎执行"""
            for req_id, prompt in zip(request_ids, request_prompts): # 获得请求ID和输入(占位符和输入文字、指令信息)
                prompt_modalities = prompt.get("modalities", None) if isinstance(prompt, dict) else None # modalities 用来告诉框架这个请求包含哪些模态
                final_stage_id = self._compute_final_stage_id(prompt_modalities) # 请求的最终stage，决定在pipline的哪一个stage结束
                final_output_stage_ids = self._compute_final_output_stage_ids(prompt_modalities) or [final_stage_id] # 最终输出从哪些 stage 取，如[1] 只取音频
                req_final_stage_ids[req_id] = final_stage_id

                # 做性能统计，不是业务处理，如stage花了多少时间，吞吐量是多少
                metrics = OrchestratorMetrics( 
                    self.num_stages,
                    self.log_stats,
                    wall_start_ts,
                    final_stage_id,
                )

                # 创建请求状态，追踪请求信息
                req_state = ClientRequestState(req_id) # ClientRequestState 用于“跟踪”记录相关信息的类
                req_state.metrics = metrics
                self.request_states[req_id] = req_state # 继承omni-base从请求ID到请求状态的映射，方便查找

                # PD disaggregation: modify stage-0 (prefill) sampling params per request
                req_sp_list = list(sampling_params_list)

                # 检查是否启用了 Prefill-Decode（PD）分离架构(传统模式 下 Prefill 和 Decode 在同一个 stage 顺序执行，共享同一组采样参数)
                pd_pair = self._get_pd_separation_pair() 
                if pd_pair is not None:
                    p_id = pd_pair[0]
                    req_sp_list[p_id] = self._prepare_prefill_sampling_params(req_id, req_sp_list[p_id])

                """ 添加请求并进行处理(后面只是进行轮询等待不断调用try_get_output() 取回各 stage 已产生的结果) """
                self.engine.add_request(
                    request_id=req_id,
                    prompt=prompt,
                    sampling_params_list=req_sp_list,
                    final_stage_id=final_stage_id,
                    final_output_stage_ids=final_output_stage_ids,
                )
                submit_ts = time.time()
                req_state.metrics.stage_first_ts[0] = submit_ts
                req_start_ts[req_id] = submit_ts

            active_reqs = set(request_ids) # 记录还没有处理完成的请求(集合移除速度快)

            # 创建进度条
            pbar = None
            if use_tqdm: 
                tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
                pbar = tqdm_func(total=len(request_ids), desc="Processed prompts", dynamic_ncols=True)

            """3. 轮询获取结果 """
            while active_reqs:
                msg = self.engine.try_get_output() # 每个stage各自一个进程进行异步处理，期间不断询问，一有结果就获取；msg既有请求ID，也有stageID用于区分请求和阶段

                should_continue, req_id, stage_id, req_state = self._handle_output_message(msg) # 从引擎消息里提取"谁发的、哪个阶段发的、要不要继续处理
                if should_continue:
                    continue

                # 检查
                if req_id not in active_reqs:
                    logger.warning("[Omni] Received output for unknown/finished request_id=%s", req_id)
                    continue
                self._check_engine_output_error(msg, req_id, stage_id) # 检查子进程执行是否出错
                if req_state.metrics is None:
                    continue

                # 把引擎返回的原始消息加工成用户能用的输出(OmniRequestOutput用户友好形式)；性能统计；`stage_id` 判断这个输出是否应该返回给用户
                output_to_yield = self._process_single_result( 
                    result=msg,
                    stage_id=stage_id,
                    metrics=req_state.metrics,
                    req_start_ts=req_start_ts,
                    wall_start_ts=wall_start_ts,
                    final_stage_id_for_e2e=req_final_stage_ids[req_id],
                )

                # 返回请求执行结果
                if output_to_yield is not None:
                    yield output_to_yield # 可以直接理解为返回结果

                # 请求完成后的收尾清
                if isinstance(msg, OutputMessage) and msg.finished:
                    active_reqs.discard(req_id) # 在剩余请求中清除已经完成的请求
                    if pbar is not None: 
                        pbar.update(1) # 进度条往前走一格
                    self._log_summary_and_cleanup(req_id) # 记录统计日志，清理该请求的状态数据（释放内存）

        except GeneratorExit:
            if "active_reqs" in locals() and active_reqs:
                self.abort(list(active_reqs))
            raise

        except Exception:
            if "active_reqs" in locals() and active_reqs:
                for req_id in active_reqs:
                    self._record_request_failure_once(req_id, reason="stage_error")
                self.abort(list(active_reqs))
            raise

        finally:
            if "pbar" in locals() and pbar is not None:
                pbar.close()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        self.engine.abort(request_ids)
        for req_id in request_ids:
            self._record_request_failure_once(req_id, reason="client_abort")
            self.request_states.pop(req_id, None)
        if self.log_stats:
            logger.info("[Omni] Aborted request(s) %s", ",".join(request_ids))
