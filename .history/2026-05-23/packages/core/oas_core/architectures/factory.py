"""Spec validator + parameter estimator + torch nn.Module builder.

The builders are intentionally compact reference implementations — Llama-style
transformer (RMSNorm + RoPE + SwiGLU + GQA), Conformer-style ASR encoder over
a transformer decoder, and a Tacotron-2-style TTS acoustic model. They are
not state-of-the-art on their own; the point is that the studio can train
*any* spec the user composes in the UI.
"""
# ruff: noqa: N806, N812
# N806/N812: this module follows standard ML tensor notation (B = batch,
# T = time/seq, L = num_layers, F = torch.nn.functional). Renaming would
# diverge from every PyTorch tutorial and paper this code is built against.

from __future__ import annotations

from typing import Any

REQUIRED_KEYS = {"modality", "hidden_size", "num_layers", "num_heads"}


def validate_spec(spec: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(spec)
    if missing:
        raise ValueError(f"spec missing keys: {sorted(missing)}")
    modality = spec["modality"]
    if modality not in ("llm", "asr", "tts"):
        raise ValueError(f"unsupported modality {modality!r}")
    if modality == "llm" and "vocab_size" not in spec:
        raise ValueError("llm spec requires vocab_size")
    h = spec["hidden_size"]
    nh = spec["num_heads"]
    if h % nh != 0:
        raise ValueError(f"hidden_size ({h}) must be divisible by num_heads ({nh})")
    nkv = spec.get("num_kv_heads", nh)
    if nh % nkv != 0:
        raise ValueError(f"num_heads ({nh}) must be divisible by num_kv_heads ({nkv})")


def estimate_params(spec: dict[str, Any]) -> int:
    """Cheap closed-form parameter estimate. Off by a few % vs the real total
    (omits biases, norm scales, embedding for ASR mel head). Good enough for
    the UI's 'about N M params' display."""
    h = spec["hidden_size"]
    inter = spec.get("intermediate_size", 4 * h)
    L = spec["num_layers"]
    nh = spec["num_heads"]
    nkv = spec.get("num_kv_heads", nh)
    head_dim = h // nh

    # Self-attention: Q (h*h) + K/V (h * nkv*head_dim each) + O (h*h)
    attn = h * h + 2 * (h * nkv * head_dim) + h * h
    # SwiGLU MLP: gate + up + down each (h*inter)
    mlp = 3 * h * inter
    per_layer = attn + mlp + 2 * h  # plus 2 RMSNorm scales

    body = per_layer * L

    if spec["modality"] == "llm":
        vocab = spec["vocab_size"]
        embed = vocab * h
        tied = spec.get("tie_embeddings", True)
        head = 0 if tied else vocab * h
        return int(body + embed + head + h)  # final norm

    if spec["modality"] == "asr":
        enc_cfg = spec.get("encoder", {})
        n_mels = enc_cfg.get("n_mels", 80)
        enc_layers = enc_cfg.get("conformer_layers", L)
        # Mel projection + conformer body (~ same per-layer cost as a transformer block)
        mel_proj = n_mels * h
        enc_body = per_layer * enc_layers
        # Decoder vocab fallbacks to LLM vocab if present, else 256.
        vocab = spec.get("vocab_size", 256)
        return int(body + enc_body + mel_proj + vocab * h + h)

    if spec["modality"] == "tts":
        tts_cfg = spec.get("tts", {})
        n_mels = tts_cfg.get("n_mels", 80)
        # Encoder for text + decoder for mel; estimate as 2 * body + mel head
        text_vocab = spec.get("vocab_size", 256)
        return int(2 * body + text_vocab * h + n_mels * h + h)

    raise ValueError(f"unsupported modality {spec['modality']!r}")


# ---------------------------------------------------------------------------
# Builders — lazy torch import.
# ---------------------------------------------------------------------------


def build_from_spec(spec: dict[str, Any]) -> Any:
    """Return an `nn.Module` built from `spec`.

    Tiny, readable reference architectures — the goal is a working forward +
    loss so the pretrain handlers can drive a real optimizer.
    """
    validate_spec(spec)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    h = spec["hidden_size"]
    inter = spec.get("intermediate_size", 4 * h)
    L = spec["num_layers"]
    nh = spec["num_heads"]
    nkv = spec.get("num_kv_heads", nh)
    head_dim = h // nh
    rope_theta = spec.get("rope_theta", 10000.0)

    class RMSNorm(nn.Module):  # type: ignore[misc]
        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: Any) -> Any:
            rms = x.pow(2).mean(-1, keepdim=True).clamp_min(self.eps).rsqrt()
            return (x * rms) * self.scale

    def _rope_cache(seqlen: int, dim: int, device: Any, dtype: Any) -> tuple[Any, Any]:
        freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seqlen, device=device).float()
        emb = torch.einsum("i,j->ij", t, freqs)
        return torch.cos(emb).to(dtype), torch.sin(emb).to(dtype)

    def _apply_rope(x: Any, cos: Any, sin: Any) -> Any:
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos.repeat_interleave(2, dim=-1)) + (rot * sin.repeat_interleave(2, dim=-1))

    class Attention(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(h, nh * head_dim, bias=False)
            self.k = nn.Linear(h, nkv * head_dim, bias=False)
            self.v = nn.Linear(h, nkv * head_dim, bias=False)
            self.o = nn.Linear(nh * head_dim, h, bias=False)

        def forward(self, x: Any, cos: Any, sin: Any, mask: Any = None) -> Any:
            B, T, _ = x.shape
            q = self.q(x).view(B, T, nh, head_dim).transpose(1, 2)
            k = self.k(x).view(B, T, nkv, head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, nkv, head_dim).transpose(1, 2)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            if nkv != nh:
                rep = nh // nkv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = attn.transpose(1, 2).reshape(B, T, nh * head_dim)
            return self.o(out)

    class SwiGLU(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(h, inter, bias=False)
            self.up = nn.Linear(h, inter, bias=False)
            self.down = nn.Linear(inter, h, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down(F.silu(self.gate(x)) * self.up(x))

    class Block(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = RMSNorm(h)
            self.attn = Attention()
            self.norm2 = RMSNorm(h)
            self.mlp = SwiGLU()

        def forward(self, x: Any, cos: Any, sin: Any) -> Any:
            x = x + self.attn(self.norm1(x), cos, sin)
            return x + self.mlp(self.norm2(x))

    # --- modality wrappers ---

    modality = spec["modality"]
    if modality == "llm":
        vocab = spec["vocab_size"]
        tied = spec.get("tie_embeddings", True)

        class LLMModel(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab, h)
                self.blocks = nn.ModuleList([Block() for _ in range(L)])
                self.norm = RMSNorm(h)
                self.head = (
                    None if tied else nn.Linear(h, vocab, bias=False)
                )

            def forward(self, input_ids: Any, labels: Any = None) -> Any:
                x = self.embed(input_ids)
                cos, sin = _rope_cache(x.shape[1], head_dim, x.device, x.dtype)
                for blk in self.blocks:
                    x = blk(x, cos, sin)
                x = self.norm(x)
                # When `tied`, head is None and we project via embed weights;
                # when not tied, `self.head` is a real nn.Linear — narrow with assert.
                if tied:
                    logits = self.embed.weight @ x.transpose(-1, -2)
                else:
                    assert self.head is not None
                    logits = self.head(x)
                if tied:
                    logits = logits.transpose(-1, -2)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, vocab), shift_labels.view(-1), ignore_index=-100
                    )
                return {"logits": logits, "loss": loss}

        return LLMModel()

    if modality == "asr":
        enc_cfg = spec.get("encoder", {})
        n_mels = enc_cfg.get("n_mels", 80)
        enc_layers = enc_cfg.get("conformer_layers", L)
        vocab = spec.get("vocab_size", 256)

        class ASRModel(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.mel_in = nn.Linear(n_mels, h)
                self.encoder = nn.ModuleList([Block() for _ in range(enc_layers)])
                self.enc_norm = RMSNorm(h)
                self.head = nn.Linear(h, vocab, bias=False)

            def forward(self, mel: Any, targets: Any = None) -> Any:
                x = self.mel_in(mel)
                cos, sin = _rope_cache(x.shape[1], head_dim, x.device, x.dtype)
                for blk in self.encoder:
                    x = blk(x, cos, sin)
                x = self.enc_norm(x)
                logits = self.head(x)
                loss = None
                if targets is not None:
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # T,B,V
                    input_lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)
                    target_lengths = (targets != 0).sum(-1)
                    loss = F.ctc_loss(
                        log_probs, targets, input_lengths, target_lengths,
                        blank=0, zero_infinity=True,
                    )
                return {"logits": logits, "loss": loss}

        return ASRModel()

    if modality == "tts":
        tts_cfg = spec.get("tts", {})
        n_mels = tts_cfg.get("n_mels", 80)
        text_vocab = spec.get("vocab_size", 256)

        class TTSModel(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.text_embed = nn.Embedding(text_vocab, h)
                self.encoder = nn.ModuleList([Block() for _ in range(L)])
                self.decoder = nn.ModuleList([Block() for _ in range(L)])
                self.mel_head = nn.Linear(h, n_mels, bias=True)

            def forward(self, text_ids: Any, mel_target: Any = None) -> Any:
                x = self.text_embed(text_ids)
                cos, sin = _rope_cache(x.shape[1], head_dim, x.device, x.dtype)
                for blk in self.encoder:
                    x = blk(x, cos, sin)
                for blk in self.decoder:
                    x = blk(x, cos, sin)
                mel_pred = self.mel_head(x)
                loss = None
                if mel_target is not None:
                    T = min(mel_pred.shape[1], mel_target.shape[1])
                    loss = F.l1_loss(mel_pred[:, :T], mel_target[:, :T])
                return {"mel": mel_pred, "loss": loss}

        return TTSModel()

    raise ValueError(f"unsupported modality {modality!r}")
