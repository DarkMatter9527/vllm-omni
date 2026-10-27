# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS talker request state with Ascend V2 attention metadata."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class NPUQwen3TTSTalkerModelState(OmniModelState, AscendModelState):
    """Talker request state whose attention metadata is the Ascend V2 one.

    MRO: Omni -> Ascend -> Default. Slots, preprocess and postprocess stay on
    the Omni lifecycle; ``prepare_attn`` (and the Ascend initializer) falls
    through to ``AscendModelState``, which reads ``attn_state`` and
    ``seq_lens_np`` off the Ascend input batch to build the attention metadata.
    """

    uses_ascend_attn_metadata = True
