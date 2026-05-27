# Roadmap: Open Audio Studio — A UI-Based Fine-Tuning & Research Platform for Speech AI

## Vision

Build an **end-to-end, UI-driven studio** where any developer can fine-tune,
full-tune, evaluate, and ship state-of-the-art **ASR, LLM, TTS, and
speech-to-speech (S2S)** models — and design entirely new architectures on top
of **JAX and PyTorch**, including **custom CUDA kernels** — without leaving the
app.

The product target is "Hugging Face AutoTrain + Weights & Biases + LM Studio +
Gradio + a kernel playground" focused on **conversational audio AI**, with
first-class support for building Gemini Live-style realtime systems.

Primary users:

- Applied ML engineers fine-tuning speech models for a product.
- Researchers prototyping new audio architectures and kernels.
- Data teams curating speech and dialog datasets.
- Solo developers who want a local, open, all-in-one alternative to closed
  studios.

Design principles:

- **UI-first, code-optional.** Every action is reachable from the UI and from a
  scriptable SDK. Nothing is UI-only or CLI-only.
- **Local by default, cloud when needed.** Runs on a single workstation; scales
  to multi-GPU clusters through the same interface.
- **Reproducible.** Every run captures dataset version, config, code commit,
  hardware, and metrics.
- **Open formats.** Datasets, checkpoints, and configs use portable formats
  (Parquet, WebDataset, safetensors, HF-compatible).
- **Composable.** ASR, LLM, TTS, and S2S modules can be mixed, swapped, and
  chained inside the same project.

## Top-Level App Structure

The studio is a single app with six primary sections, plus shared
infrastructure (jobs, experiments, models, settings).

```text
Open Audio Studio
├── ASR Studio
├── LLM Studio
├── TTS Studio
├── Speech-to-Speech (S2S) Studio
├── Datasets
└── Architecture Lab (JAX + PyTorch + CUDA)

Shared:
├── Jobs & Runs       (training, eval, inference queue)
├── Experiments       (metrics, comparisons, lineage)
├── Model Registry    (versions, tags, deploy targets)
├── Compute           (local GPUs, remote nodes, schedulers)
└── Settings          (HF token, W&B, storage, secrets)
```

## Section 1 — ASR Studio

Goal: fine-tune or fully train speech recognition models with a clean UI for
data, training, evaluation, and deployment.

Supported model families (initial):

- Whisper / faster-whisper / distil-whisper
- NVIDIA Parakeet, Canary, NeMo Conformer/Citrinet
- wav2vec2, HuBERT, WavLM
- SeamlessM4T (ASR mode)
- Bring-your-own checkpoint (HF or local)

Features:

- **Model picker** with size, languages, license, streaming support, and
  benchmark badges.
- **Training modes:** LoRA, QLoRA, adapter, partial freeze, full fine-tune,
  from-scratch.
- **Streaming vs. offline** toggle with chunk size, lookahead, and endpointing
  controls.
- **Augmentation panel:** noise, reverb, RIR, speed perturb, SpecAugment, room
  simulation, codec simulation (Opus, AMR), packet loss.
- **Tokenizer / vocab tools:** train BPE / unigram, add domain words, merge
  vocabularies, pronunciation lexicons.
- **Language and dialect targeting** with per-language data weighting.
- **Live training dashboard:** loss curves, WER/CER on validation, sample
  predictions, gradient norms, learning rate schedule.
- **Evaluation harness:**
  - WER, CER, MER, WIL overall and sliced by language, accent, domain, noise,
    SNR, device.
  - Partial-transcript stability for streaming models.
  - Endpointing latency and false-trigger rate.
  - p50/p95/p99 RTF (real-time factor) and first-token latency.
  - Confidence calibration plots.
- **Error explorer:** filter by WER, click to hear audio, see diff vs. reference,
  add to a "hard examples" set with one click.
- **Export targets:** HF format, ONNX, CTranslate2, TensorRT-LLM, OpenVINO,
  CoreML, GGML / whisper.cpp, plain safetensors.
- **One-click serve:** local HTTP / WebSocket endpoint with streaming partials.

## Section 2 — LLM Studio

Goal: fine-tune dialog and tool-using LLMs that are robust to ASR transcripts
and produce concise, spoken-style responses.

Supported model families (initial):

- Llama 3.x / 4.x
- Qwen 2.5 / 3
- Gemma 2 / 3
- Mistral, Mixtral
- Phi
- Bring-your-own HF checkpoint

Features:

- **Training modes:** LoRA, QLoRA, DoRA, GaLore, full fine-tune, continued
  pretraining, distillation.
- **Alignment modes:** SFT, DPO, ORPO, KTO, RLAIF, reward-model training.
- **Quantization:** bnb 4/8-bit, GPTQ, AWQ, FP8, MXFP4, with calibration data
  picker.
- **Context-length tools:** RoPE scaling, YaRN, position-interpolation, with
  evaluation on long-context tasks.
- **Chat template editor** with live preview and per-dataset overrides.
- **Tool-use trainer:** define tool schemas in the UI, generate synthetic
  tool-call traces, validate JSON-schema correctness during training.
- **Spoken-style trainer:** length penalties, "TTS-friendly" filters (no
  markdown / no code blocks / no emoji), prosody-aware tokens.
- **ASR-robustness trainer:** inject realistic ASR errors using a chosen ASR
  model from ASR Studio.
- **Safety pack:** refusal sets, jailbreak tests, red-team prompts, toxicity
  classifiers.
- **Evaluation harness:**
  - Task success rate, tool-selection and tool-argument accuracy.
  - Hallucination tests, faithfulness, citation tests.
  - Response length distribution and spoken-readability score.
  - MT-Bench / AlpacaEval / IFEval / GSM8K style benches, pluggable.
  - Side-by-side compare runs with blind voting.
- **Export targets:** HF, GGUF / llama.cpp, MLX, TensorRT-LLM, vLLM,
  ExecuTorch.
- **One-click serve:** OpenAI-compatible API with streaming.

## Section 3 — TTS Studio

Goal: fine-tune natural, low-latency speech synthesis aligned with the LLM's
output style.

Supported model families (initial):

- Piper
- XTTS v2 / Coqui
- StyleTTS2, VITS, Matcha-TTS, VITS2
- Parler-TTS
- F5-TTS, E2-TTS
- Bark (research mode)
- Bring-your-own checkpoint

Features:

- **Voice manager:** voice cards with speaker ID, license, consent record,
  sample audio, training stats.
- **Training modes:** speaker fine-tune, multi-speaker, voice cloning (with
  consent gate), full fine-tune, distillation to a smaller streaming model.
- **Style controls:** emotion, pacing, energy, pitch, speaking rate, accent.
- **Pronunciation tools:** lexicon editor, IPA/phoneme inspector, G2P
  fine-tuning, domain-word capture from LLM transcripts.
- **Streaming optimizer:** measure first-audio latency, RTF, chunk size sweeps.
- **Data quality tools:** SNR filter, clipping detector, alignment scorer,
  forced alignment with Montreal Forced Aligner / WhisperX.
- **Evaluation harness:**
  - Intelligibility via round-trip ASR WER.
  - Speaker similarity (ECAPA-TDNN / WavLM embeddings).
  - Naturalness MOS estimators (UTMOS, NISQA) and human MOS UI.
  - Pronunciation accuracy for a watchlist of names and terms.
  - Artifact, clipping, and long-silence detectors.
- **Consent and licensing gate:** voice cloning requires uploaded consent form
  metadata; locked behind an explicit toggle.
- **Export targets:** HF, ONNX, Piper format, TorchScript, streaming server.
- **One-click serve:** streaming WebSocket TTS endpoint.

## Section 4 — Speech-to-Speech (S2S) Studio

Goal: build, fine-tune, and evaluate full realtime voice assistants — both the
**pipeline approach** (ASR + LLM + TTS) and the **native audio approach**
(audio-token multimodal models).

### 4a. Pipeline S2S

- **Graph builder:** pick an ASR, an LLM, and a TTS from the registry, wire them
  with a visual graph, set buffering and barge-in policies.
- **Realtime playground:** browser mic + speaker, push-to-talk and
  open-mic modes, live partial transcripts, token stream, audio waveform, and
  stage-by-stage latency overlay.
- **Turn-taking controls:** VAD model, endpointing thresholds, interruption
  sensitivity, half-duplex vs. full-duplex.
- **Tool execution panel:** register tools, watch calls and results in realtime.
- **Memory panel:** short-term, long-term, and per-user memory with summarizers.

### 4b. Native Audio S2S

Supported families (as they mature):

- Moshi / Mimi codec
- Qwen2.5-Omni / Qwen3-Omni
- GLM-4-Voice
- LLaMA-Omni
- Step-Audio
- SpiritLM
- Bring-your-own audio-LM

Features:

- **Audio tokenizer training:** train or fine-tune neural codecs (EnCodec,
  Mimi, DAC, SoundStream-style).
- **Audio-LM trainer:** pretrain and SFT on interleaved audio-text tokens,
  with streaming-aware objectives.
- **Duplex training:** train models that can listen and speak simultaneously.
- **Voice preservation:** keep speaker identity across turns.

### 4c. Realtime Evaluation Harness

- Time-to-first-audio (TTFA), time-to-stop-on-interrupt.
- Per-stage p50/p95/p99 latency.
- Turn success rate, tool failure rate, transcript correction rate.
- Conversation-level evals: multi-turn task suites, barge-in scenarios,
  background-noise scenarios.
- Human listening tests with blind A/B and ranked voting.

### 4d. Deployment

- One-click deploy as a local WebSocket server, a WebRTC server, or a Docker
  container. Optional cloud targets (Modal, RunPod, Fly, k8s) via adapters.

## Section 5 — Datasets

Goal: a unified data layer for **pulling, building, curating, and versioning**
datasets for ASR, LLM, TTS, and S2S.

### 5a. Pulling Datasets

- **Hugging Face browser** with filters by modality, language, license, size,
  task. One-click import with streaming or full download.
- **Other sources:** OpenSLR, Common Voice, LibriSpeech / LibriTTS, GigaSpeech,
  People's Speech, VoxPopuli, AISHELL, MLS, Emilia, plus arbitrary HTTP / S3 /
  GCS / local folders.
- **Resumable, deduplicated downloads** with checksum verification.

### 5b. Creating Custom Datasets

Each task has a guided builder:

- **ASR dataset builder**
  - Upload audio (single file, folder, zip, URL list, RSS, YouTube with license
    gate).
  - Auto-segment with VAD, auto-transcribe with a chosen ASR for bootstrapping.
  - Human correction UI: waveform + spectrogram + editable transcript, hotkeys,
    speaker labels.
  - Quality filters: SNR, clipping, duration, language detection, alignment
    confidence.
- **TTS dataset builder**
  - Voice intake with consent form upload.
  - Auto-segmentation and forced alignment.
  - Prosody and emotion labeling.
  - Pronunciation review for OOV terms.
  - Style and speaker metadata.
- **LLM dataset builder**
  - Text import (jsonl, csv, parquet, markdown, web scrape with robots.txt
    respect).
  - Conversation builder UI with system prompts, tools, and multi-turn editor.
  - Synthetic data generation using a chosen LLM, with diversity controls.
  - ASR-noisification: run text through TTS then ASR to create realistic noisy
    transcripts.
  - Safety and refusal example builder.
- **Speech-to-speech dataset builder** (Gemini Live-style)
  - Record full conversations with timestamps, speaker turns, interruptions,
    background events.
  - Capture user audio, assistant audio, transcripts, tool calls, tool results,
    and memory snapshots in a single aligned record.
  - Support synthetic conversation generation: LLM writes a script, TTS speaks
    both sides, ASR re-transcribes, all aligned.
  - Barge-in and overlap scenarios as first-class data.

### 5c. Curation and Quality

- **Dedup** (text n-gram, audio fingerprint).
- **PII redaction** for text and audio (named-entity, phone, email, address;
  speaker anonymization options).
- **Language and quality classifiers** with bulk filtering.
- **Active learning loop:** surface low-confidence model outputs for human
  labeling.
- **Reviewer roles and assignments** for team curation.

### 5d. Versioning and Storage

- **Dataset versions** with semantic tags, diff view, and full lineage to
  training runs.
- **Manifest format:** open JSON/Parquet schema with audio path, transcript,
  language, speaker, license, splits, and arbitrary metadata.
- **Splits:** speaker-disjoint by default for audio, with stratified options.
- **Storage backends:** local FS, S3, GCS, Azure, WebDataset shards.
- **License and consent registry** attached to every sample.

## Section 6 — Architecture Lab (JAX + PyTorch + CUDA)

Goal: a workbench for inventing, profiling, and shipping **new model
architectures and custom kernels** for speech and audio AI.

### 6a. Model Authoring

- **Dual backends:** PyTorch and JAX (Flax / NNX / Equinox), with a shared
  module spec so the same architecture can be expressed in either.
- **Block library:** attention variants (MHA, MQA, GQA, MLA, sliding window,
  ring), state-space layers (Mamba, Mamba-2, RWKV), MoE routers, RoPE/ALiBi
  variants, audio-specific blocks (Conformer, FSMN, Squeezeformer), neural
  codec blocks.
- **Visual graph editor** for composing blocks, with code view and round-trip
  editing.
- **Spec presets** for ASR encoders, LLM decoders, TTS acoustic and vocoder
  models, and audio-LM stacks.
- **Parameter and FLOP estimator** with memory and KV-cache projections.

### 6b. Custom CUDA / Triton / Pallas Kernels

- **Kernel editor** with syntax highlighting and templates for:
  - Raw CUDA C++ / CUTLASS
  - Triton
  - JAX Pallas (GPU and TPU)
  - PyTorch `torch.compile` Inductor lowerings
- **Build and bind** kernels to PyTorch and JAX with one click; auto-generated
  Python bindings and gradient registration.
- **Numerical correctness harness:** compare against a reference
  PyTorch/NumPy implementation across shapes and dtypes, with tolerance
  reporting.
- **Microbenchmark harness:** sweep shapes, dtypes, and tile sizes; record
  TFLOPs, bandwidth, occupancy.
- **Profiler integration:** Nsight Systems, Nsight Compute, PyTorch profiler,
  JAX profiler, with flame graphs in the UI.
- **Autotuning:** Triton autotune presets, Pallas block-size sweeps, cache of
  best configs per GPU SKU.
- **Kernel registry:** versioned, signed, with target architectures (SM 80, 89,
  90, 100) and fallbacks.

### 6c. Training Infrastructure

- **Distributed strategies:** DDP, FSDP, FSDP2, DeepSpeed Zero 1/2/3, Megatron
  TP/PP/SP, JAX pjit / shard_map, Pathways-style multi-host.
- **Mixed precision:** fp32, tf32, bf16, fp16, fp8, mxfp4, with loss-scaling
  diagnostics.
- **Checkpointing:** sharded safetensors, async save, resumable runs, EMA.
- **Gradient tricks:** accumulation, checkpointing, selective activation
  recompute, sequence packing.
- **Data loaders:** WebDataset, Parquet, Arrow, streaming HF datasets, with
  audio-aware bucketing.

### 6d. Research Loop

- **Sweeps:** hyperparameter search with Optuna / Ax / Bayesian backends.
- **Ablation runner:** declarative ablation matrix, auto-generated comparison
  reports.
- **Notebook bridge:** open any run in a Jupyter / Marimo notebook with the
  exact env and checkpoint preloaded.
- **Paper-mode export:** auto-generate a report with config, dataset, metrics,
  and plots.

## Cross-Cutting Systems

These power every section.

### Jobs and Compute

- Local single-GPU, local multi-GPU, SSH nodes, Slurm, Kubernetes, Ray, Modal,
  RunPod, Lambda.
- Queue with priorities, GPU reservations, and spot/preemption handling.
- Live logs, TensorBoard, W&B, MLflow, and built-in experiment tracker.

### Experiments and Registry

- Every run is reproducible: code commit, container image hash, dataset version,
  config, hardware, seed.
- Model registry with stages (dev, staging, prod), tags, and signed artifacts.
- Side-by-side run comparison with metric diffs and sample-level diffs.

### Serving and Deployment

- One-click local servers per modality.
- Built-in OpenAI-compatible LLM API, WebSocket streaming ASR/TTS, WebRTC
  realtime S2S.
- Container builder for Docker / OCI with CUDA, ROCm, and CPU variants.
- Cloud deploy adapters for Modal, RunPod, Fly, k8s, SageMaker, Vertex.

### Observability

- Per-stage latency, error, and quality dashboards.
- Token, audio-second, and GPU-hour cost accounting.
- Alerting hooks (Slack, webhook, email).

### Security, Licensing, and Consent

- HF token, cloud creds, and API keys stored in OS keychain.
- License and consent metadata required for voice cloning and certain datasets.
- Audit log of training data used in each model version.
- PII redaction policies enforceable per project.

### Plugin SDK

- Python SDK that mirrors every UI action.
- Plugin points for: model families, dataset sources, augmentations, evaluators,
  exporters, deployment targets, kernels.
- Plugins are discoverable, versioned, and sandboxed.

## Tech Stack (proposed)

- **Frontend:** Next.js or SvelteKit, Tailwind, shadcn/ui, Monaco editor,
  WaveSurfer for audio, React Flow for graph editors.
- **Backend:** FastAPI + Uvicorn, WebSocket and WebRTC (aiortc / livekit),
  Postgres for metadata, Redis for queues, MinIO/S3 for artifacts.
- **ML core:** PyTorch, JAX, Hugging Face Transformers / Datasets /
  Accelerate / PEFT, vLLM, Triton, NeMo (optional adapter), torchaudio,
  soundfile.
- **Job orchestration:** Ray + a thin scheduler abstraction with Slurm and
  Kubernetes backends.
- **Tracking:** built-in store with optional W&B / MLflow mirrors.
- **Packaging:** single `pip install` or single Docker image; desktop wrapper
  via Tauri or Electron later.

## Phased Delivery

The roadmap below sequences delivery so each phase ships a usable product.

### Phase 0 — Foundation (Weeks 0–4)

- Monorepo layout: `apps/web`, `apps/server`, `packages/sdk`, `packages/core`,
  `kernels/`, `plugins/`, `infra/`.
- App shell with navigation for the six sections.
- Settings, secrets, and HF/W&B integration.
- Job queue, experiment store, model registry skeletons.
- Dataset manifest schema v1 and storage layer.
- Local GPU detection and capability report.

Exit: empty studio boots, can run a hello-world fine-tune end-to-end with a
placeholder model.

### Phase 1 — Datasets MVP (Weeks 4–8)

- HF dataset browser and importer.
- ASR, TTS, and LLM dataset builders with manual upload + auto-segment +
  auto-transcribe (Whisper baseline).
- Manifest viewer, splits, dedup, quality filters.
- Versioning and lineage.

Exit: a developer can create or import a real ASR dataset and a real LLM
dataset entirely from the UI.

### Phase 2 — ASR Studio (Weeks 8–14)

- Whisper, faster-whisper, wav2vec2 fine-tune flows (LoRA + full).
- Augmentation panel.
- Live training dashboard.
- Evaluation harness with WER slices and error explorer.
- Exporters and one-click serve.

Exit: fine-tune Whisper on a custom dataset, evaluate, export, and serve
streaming partials from the UI.

### Phase 3 — LLM Studio (Weeks 12–20)

- Llama / Qwen / Gemma fine-tune flows (LoRA, QLoRA, full).
- Chat template editor and tool-use trainer.
- ASR-robustness and spoken-style trainers (depends on Phase 2).
- Evaluation suite, side-by-side compare.
- vLLM serve and GGUF export.

Exit: fine-tune an LLM for spoken dialog with tool use and serve an
OpenAI-compatible API.

### Phase 4 — TTS Studio (Weeks 18–26)

- Piper, XTTS, StyleTTS2, F5-TTS fine-tune flows.
- Voice manager with consent gating.
- Pronunciation tools and streaming optimizer.
- Evaluation harness (UTMOS/NISQA, round-trip ASR, MOS UI).
- Streaming TTS server.

Exit: fine-tune a custom voice and serve streaming TTS with measurable
first-audio latency.

### Phase 5 — S2S Pipeline Studio (Weeks 24–32)

- Visual graph builder for ASR + LLM + TTS.
- Realtime browser playground with VAD, barge-in, turn IDs.
- Tool execution and memory panels.
- Realtime evaluation harness (TTFA, interrupt latency, per-stage p95).
- WebSocket and WebRTC deployment.

Exit: a developer can assemble a Gemini Live-style assistant in the UI and
have a real conversation with it locally.

### Phase 6 — Architecture Lab MVP (Weeks 28–38)

- Dual-backend module spec (PyTorch + JAX).
- Block library and visual graph editor.
- Triton + CUDA kernel editor with build, bind, and correctness harness.
- Microbenchmark and profiler integration.
- Distributed training presets (FSDP2, DeepSpeed, JAX pjit).

Exit: a researcher can define a new attention variant, write a Triton kernel,
verify correctness, benchmark it, and train a small model end-to-end inside
the studio.

### Phase 7 — Native Audio S2S (Weeks 36–48)

- Audio tokenizer trainer (Mimi/EnCodec/DAC).
- Audio-LM training recipes (Moshi-style, Qwen-Omni-style).
- Duplex training and voice-preserving generation.
- Comparison harness against the pipeline S2S baseline.

Exit: train and serve a native audio S2S model from the studio, side-by-side
with the pipeline version.

### Phase 8 — Collaboration and Cloud Scale (Weeks 44–56)

- Multi-user projects, roles, and reviews.
- Cloud compute adapters (Modal, RunPod, k8s, Slurm).
- Shared model registry and dataset hub.
- Audit, billing, and quota tooling.

Exit: a team can collaborate on datasets, runs, and models across local and
cloud GPUs from a single workspace.

### Phase 9 — Polish, Plugins, and Public Release (Weeks 52–64)

- Plugin SDK and public plugin registry.
- Desktop app (Tauri).
- Docs, tutorials, golden-path projects (build-your-own-Gemini-Live,
  build-your-own-Whisper, build-your-own-voice).
- Public beta.

Exit: 1.0 release.

### Phase 10 — Advanced Research & Visualization Suite (Weeks 60–72)

- **Interactive Learning Rate Scheduler Simulator & SVG Grapher**: Live knob dials to simulate Cosine Annealing, warmups, and decays with interactive SVG plots before training submission.
- **Audio Tokenizer Spectrogram Reconstruction Heatmap Analyzer**: Visual spectrogram bands analyzer to measure exact frequency-band compression loss and codebook bit-rate efficiency.
- **Visual Model Spec Node-Graph block compiler**: Dynamic block diagram builder displaying the neural dataflow (Encoders -> Attention blocks -> Decoders) compiled from JSON schemas.

Exit: World-class visualization suite for speech-model researchers and deep deep-learning engineers.

## Milestones

- **M1 — Studio shell:** app boots, six sections navigable, jobs run. `[COMPLETED]`
- **M2 — Data in:** HF import + custom ASR/LLM/TTS dataset builders. `[COMPLETED]`
- **M3 — Train ASR:** fine-tune and serve a Whisper variant from the UI. `[COMPLETED]`
- **M4 — Train LLM:** fine-tune and serve a tool-using spoken-style LLM. `[COMPLETED]`
- **M5 — Train TTS:** fine-tune and serve a streaming custom voice. `[COMPLETED]`
- **M6 — Pipeline S2S:** realtime browser conversation with barge-in. `[COMPLETED]`
- **M7 — Architecture Lab:** custom Triton/CUDA kernel + new block trained end-to-end. `[COMPLETED]`
- **M8 — Native S2S:** audio-LM trained and served from the studio. `[COMPLETED]`
- **M9 — Team and cloud:** multi-user + cloud compute. `[COMPLETED]`
- **M10 — 1.0:** plugins, desktop app, public beta. `[COMPLETED]`
- **M11 — Research Suite:** dynamic schedulers, codec spectrogram analyzers, and node-graphs operational. `[PLANNED]`

## Immediate Next Steps (Phase 10 Execution)

1. **Implement the SVG Scheduler Simulator**: Wire numerical slider forms to dynamic math functions plotting step-by-step decay graphs on the templates panel.
2. **Build the Spectrogram Heatmap canvas**: Integrate HTML5 Canvas / SVG grids displaying Mel Mel-spectral difference indices for compressed speech wave forms.
3. **Launch the Visual Spec Node-Graph compiler**: Connect JSON spec objects to beautiful hierarchical block charts mapping custom neural attention layers.
