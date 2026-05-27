# Open Audio Studio

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Node.js 18+](https://img.shields.io/badge/Node.js-18+-339933.svg?style=flat&logo=node.js&logoColor=white)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-000000.svg?style=flat&logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

An **end-to-end, UI-driven studio** where machine learning engineers and speech AI researchers can fine-tune, full-tune, evaluate, and deploy state-of-the-art **ASR, LLM, TTS, and Speech-to-Speech (S2S)** models — and design entirely new architectures on top of **JAX and PyTorch** (including compiling custom **Triton, CUDA, and JAX Pallas kernels**) — without ever leaving the web browser.

Open Audio Studio delivers the functionality of **Hugging Face AutoTrain + Weights & Biases + LM Studio + Gradio + an Interactive Kernel Playground** focused entirely on conversational voice systems.

---

## 🌟 Core Features

* **🎙️ ASR Studio (Speech-to-Text)**: Train and fine-tune Whisper, wav2vec2, and Conformer weights. Incorporates interactive audio augmentation tools (noise injection, room impulse responses, speed perturbations) and Word Error Rate (WER/CER) evaluation dashboards.
* **🧠 LLM Studio (Language Modeling)**: Autoregressively train language adapters on conversation transcripts. Optimize spoken fill-in tokens and turn-taking dialogue behaviors.
* **🔊 TTS Studio (Text-to-Speech)**: Synthesize speech from text inputs. Leverage high-fidelity acoustic encoders, flow-matching Integration synthesizers, and HiFi-GAN upsampling vocoders.
* **🔄 Speech-to-Speech (S2S) Studio**: Deploy duplex spoken agents bypassing separate cascade decodes. Seamlessly train Native Audio-LMs (such as Qwen2.5-Omni, LLaMA-Omni, or GLM-4-Voice) with overlap attention masking (for user barge-ins) and voice style preservation (ECAPA-TDNN speaker similarity matching).
* **🏗️ Architecture Lab**: An IDE for writing custom PyTorch and JAX tensor operations. Compile custom **Triton / CUDA / Pallas kernels**, verify numerical correctness against reference implementations, and autotune GPU performance grid shapes.
* **📦 Workspace & Templates Hub**: Access pre-loaded, copy-pasteable JSON recipes and guides to train, fine-tune, or compile models across all modalities from scratch.

---

## 📊 Advanced Research & Visualization Suite

Open Audio Studio includes three high-fidelity visual dashboards to aid speech researchers in diagnosing schedules, spectral compression losses, and layer connections:

1. **Interactive SVG Learning Rate Scheduler Simulator**:
   * Simulates linear warmups and cosine decay schedules before running large-scale training jobs.
   * Recalculates complex coordinates in real-time, plotting glowing neon purple paths and gradients.
2. **Audio Tokenizer Spectrogram Reconstruction Heatmap**:
   * Evaluates compression losses across quantizer stage codebooks (Mimi, EnCodec, and DAC).
   * Renders three parallel grids (Original Waveform vs. Decoded Waveform vs. Spectral Difference Heatmap) using distinct HSL color highlights, pinpointing high-frequency compression artifacts.
3. **Model Spec Node-Graph Spec Compiler**:
   * Materializes complete `ModelSpec` schemas into graphical flow diagrams (Input Modality -> Encoder Stack -> Cross-Attention Projection -> Output Vocoder/Decoder Synthesis).
   * Automatically benchmarks model parameters, layer depths, and GFLOPS occupancy.

---

## 📂 Monorepo Layout

```text
open-audio-studio/
├── apps/
│   ├── web/           # Next.js 14 React web application (TailwindCSS, Lucide Icons)
│   └── server/        # FastAPI Python backend (REST, WebSockets session handlers)
├── packages/
│   ├── core/          # Core modules: SQLAlchemy schemas, WorkerPool queue, storage, manifests
│   └── sdk/           # Python developer SDK client mirroring every studio API
├── kernels/           # Autotuned CUDA / Triton / Pallas kernel directories
├── plugins/           # Scaffolding and modules for first-party plugins
├── tests/             # Top-level integration & unit test suites (95 tests passing)
├── start_studio.sh    # Executable environment installer & local server orchestrator
├── Makefile           # Dev scripts runner (test, lint, compile, clean)
└── ROADMAP.md         # Full phased product vision document
```

---

## 🛠️ System Requirements

* **Operating System**: Linux (tested on Ubuntu 20.04/22.04) or macOS.
* **Python**: Version 3.11 or newer.
* **Node.js**: Version 18 or newer.
* **GPU (Optional)**: NVIDIA GPU with CUDA 12 support (strongly recommended for custom CUDA/Triton compilation and local training).

---

## 🚀 Quick Start (Automated Setup)

We have provided a unified shell bootstrapper to handle virtual environments, install Python dependencies, resolve Next.js requirements, and launch both services in parallel:

```bash
# 1. Clone the repository and navigate inside
git clone https://github.com/your-username/open-audio-studio.git
cd open-audio-studio

# 2. Run the bootstrapper script
./start_studio.sh
```

The script automatically releases occupied ports, activates a local `.venv`, installs all packages in editable mode, and boots the servers:
* **Next.js Web Interface**: [http://localhost:3000](http://localhost:3000)
* **FastAPI API Backend**: [http://localhost:8000](http://localhost:8000)
* **Interactive API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

*To stop both servers at once, press `Ctrl+C` in your terminal.*

---

## 🔧 Manual Setup & Alternative Runners

If you prefer to manage environments manually, you can use the provided `Makefile`:

```bash
# 1. Install all dependencies (Python editable modules + web node modules)
make install

# 2. Concurrently run the servers
make dev

# 3. Execute the full unit test suite (95 tests)
make test

# 4. Perform linting & typechecks
make lint
make typecheck
```

---

## 📖 Educational Tutorials

### Tutorial 1: Fine-Tuning Whisper ASR (Parameter-Efficient LoRA)
To adapt a pre-trained ASR model (like `openai/whisper-small`) to high-noise environments or domain terms:
1. In the **Datasets Studio**, upload your transcriptions and audio samples, then split into `train` and `val`.
2. Navigate to **Templates & Recipes**, choose **ASR (🎙️)** -> **LoRA Fine-Tuning**.
3. Copy the template JSON and navigate to **ASR Fine-Tuning** in the sidebar.
4. Adjust parameters (e.g., target attention modules `["q_proj", "v_proj"]`, learning rate `2e-4`, Rank `8`) and click **Start fine-tune**.
5. Monitor training and evaluation Word Error Rate (WER) converge in real-time.

### Tutorial 2: Creating a Custom Model Spec from Scratch
To prototype a custom Spoken LLM:
1. Open the **Architecture Lab** and select **Custom Spoken LM**.
2. Define your architecture configuration schema in the IDE spec panel:
   ```json
   {
     "slug": "custom-spoken-lm",
     "name": "Custom Spoken LM",
     "modality": "llm",
     "spec": {
       "vocab_size": 32000,
       "hidden_size": 2048,
       "num_hidden_layers": 12,
       "num_attention_heads": 16,
       "activation_function": "swiglu"
     }
   }
   ```
3. Click **Verify & Graph spec**. The compiler will parse the JSON, benchmark the GFLOPS timing curves, and output an interactive block flowchart rendering all 12 causal layers.
4. Click **Register Model** to save the architecture layout to the registry, ready for custom training runs.

---

## 📜 License

Licensed under the **Apache License, Version 2.0** (the "License"). You may not use this project except in compliance with the License. You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
