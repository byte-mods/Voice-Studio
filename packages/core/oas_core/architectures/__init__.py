"""Architecture spec → torch.nn.Module factory.

The spec is a plain JSON document the UI's block editor produces. It is
deliberately small and intuitive; a maximum-flexibility version (everything
PyTorch lets you build) is out of scope for v1. Today's vocabulary:

```
{
  "modality": "llm" | "asr" | "tts",
  "vocab_size": 32000,                # required for llm / asr-decoder
  "hidden_size": 512,
  "intermediate_size": 2048,
  "num_layers": 12,
  "num_heads": 8,
  "num_kv_heads": 8,                  # MHA vs GQA
  "max_seq_len": 2048,
  "rope_theta": 10000.0,
  "tie_embeddings": true,
  "encoder": {                        # ASR only
    "n_mels": 80,
    "conv_stride": 2,
    "conformer_layers": 12
  },
  "tts": {                            # TTS only
    "n_mels": 80,
    "n_fft": 1024,
    "sample_rate": 22050
  }
}
```

`build_from_spec` returns an `nn.Module` whose forward signature is the
appropriate one for the modality (LM gets `input_ids`, ASR gets `mel`,
TTS gets `text_ids`). The corresponding pretrain handler knows the shape.
"""

from oas_core.architectures.factory import (
    build_from_spec,
    estimate_params,
    validate_spec,
)

__all__ = ["build_from_spec", "estimate_params", "validate_spec"]
