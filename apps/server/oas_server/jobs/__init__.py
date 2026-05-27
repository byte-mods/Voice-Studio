"""Built-in job handlers.

Importing this package registers every handler.
"""

from oas_server.jobs import (
    asr_bootstrap,
    asr_eval,
    codec_train,
    hf_import,
    kernel_bench,
    llm_eval,
    llm_finetune,
    s2s_native_finetune,
    tts_eval,
    tts_finetune,
    whisper_finetune,
)

__all__ = [
    "asr_bootstrap",
    "asr_eval",
    "codec_train",
    "hf_import",
    "kernel_bench",
    "llm_eval",
    "llm_finetune",
    "s2s_native_finetune",
    "tts_eval",
    "tts_finetune",
    "whisper_finetune",
]
