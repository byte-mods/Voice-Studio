"use client";

import { useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import { Card, CardTitle } from "@/components/Card";
import {
  Mic,
  Brain,
  Speaker,
  AudioWaveform,
  BookOpen,
  Copy,
  Check,
  Cpu,
  Activity,
  Sliders,
  Settings,
  GraduationCap
} from "lucide-react";

type Modality = "asr" | "llm" | "tts" | "sts";
type Setup = "lora" | "full" | "scratch";

export default function TemplatesPage() {
  const [activeModality, setActiveModality] = useState<Modality>("asr");
  const [activeSetup, setActiveSetup] = useState<Setup>("lora");
  const [copied, setCopied] = useState(false);

  // Scheduler Simulator State
  const [peakLr, setPeakLr] = useState(5e-4);
  const [warmupSteps, setWarmupSteps] = useState(1000);
  const [totalSteps, setTotalSteps] = useState(10000);
  const [cosineCycles, setCosineCycles] = useState(1);

  // Generate SVG path coordinates dynamically
  const width = 600;
  const height = 150;
  const points: string[] = [];
  const resolution = 120;
  for (let i = 0; i <= resolution; i++) {
    const step = (i / resolution) * totalSteps;
    let lr = 0;
    if (step < warmupSteps) {
      lr = (step / warmupSteps) * peakLr;
    } else {
      const progress = (step - warmupSteps) / (totalSteps - warmupSteps);
      const angle = progress * Math.PI * cosineCycles;
      lr = 0.5 * peakLr * (1 + Math.cos(angle));
    }
    const x = (step / totalSteps) * width;
    const y = height - (lr / Math.max(1e-9, peakLr)) * (height - 30) - 15;
    points.push(`${x},${y}`);
  }
  const pathData = `M ${points.join(" L ")}`;
  const areaPathData = `${pathData} L ${width},${height - 5} L 0,${height - 5} Z`;

  // Trigger copy indicator
  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  // Multi-modality / Multi-setup content dictionary
  const templatesDb: Record<Modality, Record<Setup, { title: string; desc: string; payload: string; tutorial: string }>> = {
    asr: {
      lora: {
        title: "🎙️ ASR Whisper LoRA Fine-Tuning",
        desc: "Parameter-efficient fine-tuning configuration to adapt pre-trained Whisper weights to specific domain vocabulary, accents, or background noise profiles.",
        payload: `{
  "job_kind": "whisper_finetune",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "base_model": "openai/whisper-small",
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "peft": {
      "r": 8,
      "alpha": 32,
      "target_modules": ["q_proj", "v_proj"],
      "dropout": 0.05
    },
    "hyperparameters": {
      "learning_rate": 2e-4,
      "batch_size": 16,
      "epochs": 3,
      "warmup_steps": 100,
      "gradient_accumulation_steps": 2
    },
    "augmentations": {
      "noise_injection": true,
      "speed_perturbation": true,
      "pitch_shift": false
    }
  }
}`,
        tutorial: `### 📘 Concept: Whisper LoRA Adaptations
Adapting speech-to-text models like Whisper requires training key attention projection heads without destroying the pre-trained weights. By freezing base layers and inserting small-rank matrices ($r=8$) into query/value paths ($q_proj$, $v_proj$), we reduce parameters by **99.2%** while improving domain Word Error Rate (WER) by up to **35%**.

#### 🚀 How to Execute:
1. Copy the JSON config payload above.
2. Go to the **Jobs** page in the Workspace sidebar.
3. Click **Submit Job**, set the job type to \`whisper_finetune\`, paste your config, and click **Submit**.
4. Monitor the Word Error Rate (WER) decrease in real-time on your **Experiments** telemetry page!`
      },
      full: {
        title: "🚀 ASR End-to-End Full-Tuning",
        desc: "Full parameter optimization recipe to retrain encoder-decoder ASR sequences, maximizing cross-attention convergence on specialized high-fidelity datasets.",
        payload: `{
  "job_kind": "asr_pretrain",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "encoder_layers": 8,
    "decoder_layers": 6,
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "hyperparameters": {
      "learning_rate": 5e-5,
      "optimizer": "adamw",
      "weight_decay": 0.01,
      "batch_size": 32,
      "epochs": 10,
      "scheduler": "cosine"
    },
    "feature_extractor": {
      "num_mel_bins": 80,
      "sampling_rate": 16000
    }
  }
}`,
        tutorial: `### 📘 Concept: Full Encoder-Decoder Optimization
When adapting ASR models to entirely new languages or highly degraded audio, LoRA can sometimes restrict convergence. Full-tuning updates all weights of the audio encoder (CNN feature extractors + Transformer blocks) and language decoder, enabling the model to learn completely new acoustic features and phonetics.

#### 🚀 How to Execute:
1. Ensure your dataset is imported and split (e.g. ASR modality dataset version).
2. Launch the \`asr_pretrain\` job using the config payload.
3. Use a high-end serverless cluster adapter (e.g., Slurm or Modal H100s) as full optimization consumes substantial GPU overhead!`
      },
      scratch: {
        title: "🏗️ ASR Architecture Spec (From Scratch)",
        desc: "Architecture specification schema designed to compile a completely custom ASR neural model inside the Architecture Lab IDE.",
        payload: `{
  "slug": "custom-spectrogram-asr",
  "name": "Custom Spectrogram ASR",
  "modality": "asr",
  "spec": {
    "encoder": {
      "conv_blocks": [
        {"filters": 32, "kernel_size": 3, "stride": 2},
        {"filters": 64, "kernel_size": 3, "stride": 2}
      ],
      "transformer_layers": 6,
      "hidden_dim": 512,
      "attention_heads": 8,
      "norm_type": "rmsnorm"
    },
    "decoder": {
      "transformer_layers": 4,
      "hidden_dim": 512,
      "attention_heads": 8,
      "vocab_size": 50257
    }
  }
}`,
        tutorial: `### 📘 Concept: Designing ASR from Scratch
Using the **Architecture Lab**, you can invent completely unique ASR structures. This spec registers an encoder composed of custom convolutional downsamplers feeding into 6 Transformer layers using \`RMSNorm\`, mapped to a 4-layer auto-regressive decoder. 

#### 🚀 How to Execute:
1. Go to the **Architecture Lab** IDE.
2. Select your Custom Spectrogram ASR.
3. Paste the Spec JSON into the architecture spec panel, click **Verify & Compile** to materialize the \`torch.nn.Module\` and view GFLOPS benchmarking occupancy!`
      }
    },
    llm: {
      lora: {
        title: "🧠 LLM Spoken-Style LoRA Fine-Tuning",
        desc: "Adapts a causal LLM to learn conversational turn tokens, speech filler markers, and spoken-style response patterns.",
        payload: `{
  "job_kind": "llm_finetune",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "peft": {
      "r": 16,
      "alpha": 64,
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
      "dropout": 0.05
    },
    "hyperparameters": {
      "learning_rate": 1e-4,
      "batch_size": 8,
      "epochs": 3,
      "max_seq_length": 2048
    }
  }
}`,
        tutorial: `### 📘 Concept: Spoken-Style Adaptations
Text models are typically trained on formal articles, making them sound unnatural when read out loud by a voice synthesizer. Spoken-style fine-tuning teaches the LLM to write with conversational markers ("umm", "well", "ah") and shorter, interruption-friendly sentences, ensuring natural turn-taking dynamics.

#### 🚀 How to Execute:
1. Import a dialogue transcription dataset and split it inside the Datasets Studio.
2. Submit the \`llm_finetune\` job using the config payload.
3. Once completed, deploy the served endpoint and test the dialog flow in your S2S Playground!`
      },
      full: {
        title: "🚀 LLM Causal Pretraining",
        desc: "Full autoregressive Transformer pre-training config to optimize a custom language model from scratch on large-scale domain conversation transcripts.",
        payload: `{
  "job_kind": "pretrain",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "hyperparameters": {
      "learning_rate": 6e-4,
      "optimizer": "adamw",
      "batch_size": 64,
      "scheduler": "cosine_with_warmup",
      "warmup_steps": 1000,
      "weight_decay": 0.1
    }
  }
}`,
        tutorial: `### 📘 Concept: Autoregressive Language Modeling
Full causal pretraining optimizes all hidden attention parameters to predict the next token given preceding context. It uses Rotary Position Embeddings (RoPE) and AdamW with weight decay, optimizing sequence predictions over millions of conversation dialogue turns.

#### 🚀 How to Execute:
1. Ensure your training datasets are imported and fully tokenized.
2. Submit the \`pretrain\` job using the config.
3. Set your compute provider to a high-scale GPU cluster (e.g., Slurm SSH node or Modal) to execute multi-node distributed training.`
      },
      scratch: {
        title: "🏗️ LLM Architecture Spec (From Scratch)",
        desc: "Detailed spec layout to initialize and compile a custom casual Transformer language model from scratch inside the Architecture Lab.",
        payload: `{
  "slug": "custom-spoken-lm",
  "name": "Custom Spoken LM",
  "modality": "llm",
  "spec": {
    "vocab_size": 32000,
    "hidden_size": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "intermediate_size": 5632,
    "activation_function": "swiglu",
    "max_position_embeddings": 4096,
    "tie_word_embeddings": false
  }
}`,
        tutorial: `### 📘 Concept: Designing a Custom Spoken LM
This specification defines a 12-layer causal decoder using the \`SwiGLU\` activation function and a hidden dimension of 2048. It is optimized to map conversational tokens across a 4096 context window, utilizing tie-free embedding projections.

#### 🚀 How to Execute:
1. Go to the **Architecture Lab** IDE, select "Custom Spoken LM".
2. Paste the Spec JSON, compile, and benchmark it.
3. Register the compiled model, and feed it into the causal SFT pretraining job queue!`
      }
    },
    tts: {
      lora: {
        title: "🔊 TTS Reference Voice LoRA Adapter",
        desc: "Parameter-efficient recipe to adapt text-to-semantic projections, fine-tuning pre-trained vocal structures to match a target speaker's vocal profile.",
        payload: `{
  "job_kind": "tts_finetune",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "base_model": "tts-coqui-xtts-v2",
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "reference_audio_uri": "s3://my-bucket/voices/target-voice.wav",
    "peft": {
      "r": 4,
      "alpha": 16,
      "target_modules": ["wte", "linear_proj"]
    },
    "hyperparameters": {
      "learning_rate": 5e-5,
      "batch_size": 4,
      "epochs": 5
    }
  }
}`,
        tutorial: `### 📘 Concept: Reference Voice Adaptations
Standard text-to-speech models output default voices. To clone a specific speaker's identity (pitch, tone, prosody), we freeze the sound generator (vocoder) and adapt only the linear acoustic projection mapping matrices using small rank LoRA, matching the speaker's vocal map vector.

#### 🚀 How to Execute:
1. Upload a 30-second clean audio sample of the target voice to your storage bucket.
2. Submit the \`tts_finetune\` job using the config payload.
3. Test your newly fine-tuned voice in the serving playground to evaluate similarity metrics!`
      },
      full: {
        title: "🚀 TTS Acoustic Model Full-Tuning",
        desc: "Full optimization parameters to train a custom acoustic flow-matching or VITS acoustic synthesizer from scratch.",
        payload: `{
  "job_kind": "tts_pretrain",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "hyperparameters": {
      "learning_rate": 1e-4,
      "batch_size": 16,
      "epochs": 15,
      "optimizer": "adamw",
      "scheduler": "exponential"
    },
    "vocoder": {
      "generator_layers": 4,
      "upsample_rates": [8, 8, 2, 2]
    }
  }
}`,
        tutorial: `### 📘 Concept: Flow-Matching TTS Optimizations
Full acoustic model tuning optimizes both the semantic duration predictor and the vocoder generator (HiFi-GAN style upsampling layers). It retrains spectral spectrogram matching metrics directly, creating a robust vocoder capable of producing high-fidelity natural waveforms.

#### 🚀 How to Execute:
1. Prepare a high-fidelity single-speaker studio dataset (e.g. LJSpeech modality dataset version).
2. Launch the \`tts_pretrain\` job using the config.
3. Monitor your mel-spectral reconstruction loss on your **Experiments** panel.`
      },
      scratch: {
        title: "🏗️ TTS Flow-Matching Spec (From Scratch)",
        desc: "Architecture layout designed to register and compile a completely custom Flow-Matching Text-to-Speech synthesizer.",
        payload: `{
  "slug": "custom-flow-tts",
  "name": "Custom Flow TTS",
  "modality": "tts",
  "spec": {
    "text_encoder": {
      "transformer_layers": 4,
      "hidden_dim": 256,
      "attention_heads": 4
    },
    "flow_matching": {
      "ode_steps": 10,
      "sigma_min": 1e-4,
      "hidden_dim": 256
    },
    "vocoder": {
      "hidden_dim": 512,
      "upsample_rates": [8, 8, 4]
    }
  }
}`,
        tutorial: `### 📘 Concept: Designing Flow-Matching TTS
This spec defines a modern Flow-Matching ODE integration synthesizer. It uses a 4-layer Transformer text encoder to map characters to phonetic spaces, coupled with an Ordinary Differential Equation (ODE) solver doing 10 integration steps to reconstruct highly natural speech.

#### 🚀 How to Execute:
1. Compile the Spec JSON inside the **Architecture Lab** IDE.
2. Submit the registered custom layout into the full \`tts_pretrain\` job queue.
3. Deploy the serves and test its speech synthesis speed!`
      }
    },
    sts: {
      lora: {
        title: "🔄 STS Voice Preservation SFT",
        desc: "SFT configuration payload to fine-tune a native Audio-LM on voice preservation hooks and ECAPA-TDNN embedding target spaces.",
        payload: `{
  "job_kind": "s2s_native_finetune",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "base_model": "qwen-omni-style-audio-lm",
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "duplex_mode": true,
    "peft": {
      "r": 16,
      "alpha": 32,
      "target_modules": ["q_proj", "v_proj"]
    },
    "reference_speaker_embedding": "YOUR_ECAPA_EMBEDDING_VECTOR",
    "hyperparameters": {
      "learning_rate": 1e-4,
      "batch_size": 8,
      "epochs": 3
    }
  }
}`,
        tutorial: `### 📘 Concept: Voice-Preserved Speech SFT
Speech-to-Speech models must preserve vocal profiles during translation or conversational responses. By training the causal Audio-LM on interleaved turn-masks with an injected ECAPA-TDNN reference speaker vector, we force the output audio quantizer heads to preserve pitch, prosody, and tone.

#### 🚀 How to Execute:
1. Copy the JSON config, replacing the reference speaker vector space.
2. Go to the **ASR/S2S Fine-Tuning** panel, click **Finetune**, and enable **Voice Preservation**.
3. Once servable, test live in the Blind Arena playground to verify consistency!`
      },
      full: {
        title: "🚀 STS Native Audio-LM Full-Tuning",
        desc: "Full autoregressive Audio-LM pretraining recipe to train a joint voice-to-voice transformer model from scratch.",
        payload: `{
  "job_kind": "codec_train",
  "project_id": "YOUR_PROJECT_ID",
  "config": {
    "dataset_version_id": "YOUR_DATASET_VERSION_ID",
    "quantizer": {
      "stages": 8,
      "codebook_size": 1024,
      "dimension": 128
    },
    "hyperparameters": {
      "learning_rate": 2e-4,
      "batch_size": 32,
      "epochs": 8,
      "spectral_weight": 1.0,
      "adversarial_weight": 0.1
    }
  }
}`,
        tutorial: `### 📘 Concept: Native Multimodal Audio-LMs
Training a native STS model maps audio codes (tokens) directly. The model causal Transformer predicts subsequent audio tokens given user prompt audio tokens. This avoids any ASR or TTS bottlenecks, optimizing conversational response times under 150ms!

#### 🚀 How to Execute:
1. Ensure your speech datasets are imported and processed.
2. Submit the \`codec_train\` job using the config payload to train your RVQ codebook quantizers.
3. Serve the native model and test time-to-first-audio curves in your side-by-side arena.`
      },
      scratch: {
        title: "🏗️ STS Dual-Stream Spec (From Scratch)",
        desc: "Architecture spec design designed to compile a custom Moshi/Omni-style Dual-Stream Speech-to-Speech audio-LM.",
        payload: `{
  "slug": "custom-omni-sts",
  "name": "Custom Omni STS",
  "modality": "sts",
  "spec": {
    "audio_encoder": {
      "type": "rvq_codec",
      "codebooks": 8,
      "vocab_size": 1024
    },
    "joint_transformer": {
      "layers": 16,
      "hidden_dim": 1024,
      "attention_heads": 16,
      "norm_type": "rmsnorm"
    },
    "audio_decoder": {
      "type": "vocoder_hifigan",
      "upsample_channels": 512
    }
  }
}`,
        tutorial: `### 📘 Concept: Designing Dual-Stream STS
This specification defines a state-of-the-art dual-stream Audio-LM. It utilizes an 8-stage Residual Vector Quantizer (RVQ) codec to map raw user speech, combined with a 16-layer joint transformer that outputs synthesized response audio tokens.

#### 🚀 How to Execute:
1. Go to the **Architecture Lab** IDE, select "Custom Omni STS".
2. Paste the Spec JSON, compile, and benchmark the kernel speeds.
3. Feed the registered model into the native SFT job queue!`
      }
    }
  };

  const currentRecipe = templatesDb[activeModality][activeSetup];

  return (
    <>
      <PageHeader
        title="Templates & Recipes Library"
        subtitle="Access copy-pasteable JSON configs and educational tutorials to fine-tune, full-tune, and pre-train custom models."
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar Selector Controls */}
        <div className="space-y-4">
          <Card className="shadow-md">
            <CardTitle className="text-xs font-bold uppercase tracking-wider text-muted mb-3 flex items-center gap-1">
              <GraduationCap className="w-4 h-4 text-accent" /> Select Modality
            </CardTitle>
            <div className="space-y-1.5">
              {[
                { id: "asr", name: "🎙️ Speech-to-Text (ASR)", desc: "Whisper & Encoder-Decoder" },
                { id: "llm", name: "🧠 Language Model (LLM)", desc: "Causal Text Transformers" },
                { id: "tts", name: "🔊 Text-to-Speech (TTS)", desc: "Flow-Matching Acoustic Models" },
                { id: "sts", name: "🔄 Speech-to-Speech (STS)", desc: "Native Multimodal Audio-LM" },
              ].map((mod) => (
                <button
                  key={mod.id}
                  onClick={() => {
                    setActiveModality(mod.id as Modality);
                    setCopied(false);
                  }}
                  className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all ${
                    activeModality === mod.id
                      ? "bg-accent/10 border-accent text-accent shadow-sm"
                      : "bg-transparent border-border hover:bg-border/20 text-fg/80"
                  }`}
                >
                  <div className="font-semibold">{mod.name}</div>
                  <div className="text-[10px] text-muted mt-0.5">{mod.desc}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card className="shadow-md">
            <CardTitle className="text-xs font-bold uppercase tracking-wider text-muted mb-3 flex items-center gap-1">
              <Sliders className="w-4 h-4 text-purple-400" /> Setup Profile
            </CardTitle>
            <div className="space-y-1.5">
              {[
                { id: "lora", name: "🔧 LoRA Fine-Tuning", desc: "Parameter-efficient domain adaptation" },
                { id: "full", name: "🚀 Full-Tune / Pre-train", desc: "End-to-end full parameter training" },
                { id: "scratch", name: "🏗️ Architecture (Scratch)", desc: "Custom specs for Architecture Lab" },
              ].map((set) => (
                <button
                  key={set.id}
                  onClick={() => {
                    setActiveSetup(set.id as Setup);
                    setCopied(false);
                  }}
                  className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all ${
                    activeSetup === set.id
                      ? "bg-purple-500/10 border-purple-500/30 text-purple-300 shadow-sm"
                      : "bg-transparent border-border hover:bg-border/20 text-fg/80"
                  }`}
                >
                  <div className="font-semibold">{set.name}</div>
                  <div className="text-[10px] text-muted mt-0.5">{set.desc}</div>
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* Right Side: Code Viewer & Tutorial */}
        <div className="lg:col-span-3 space-y-6">
          {/* Header Card */}
          <Card className="shadow-md border border-border/60">
            <div className="flex items-start justify-between gap-4 border-b border-border/20 pb-3 mb-3">
              <div>
                <h2 className="text-lg font-bold text-fg/90 flex items-center gap-2">
                  {currentRecipe.title}
                </h2>
                <p className="text-xs text-muted mt-1 leading-relaxed">
                  {currentRecipe.desc}
                </p>
              </div>
              <button
                onClick={() => copyToClipboard(currentRecipe.payload)}
                className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-card/40 hover:bg-border/30 active:scale-[0.98] transition-all text-xs font-semibold text-fg/80 shadow-sm shrink-0"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" /> Copied!
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" /> Copy Payload
                  </>
                )}
              </button>
            </div>

            {/* Code Block Viewer */}
            <div className="relative border border-border/40 rounded-lg overflow-hidden bg-black/85 shadow-inner">
              <div className="bg-zinc-900/80 px-3 py-1 border-b border-border/30 flex items-center justify-between text-[10px] text-zinc-400 font-mono uppercase tracking-wider">
                recipe.json configuration template
              </div>
              <pre className="p-4 text-[11px] font-mono text-emerald-400 h-80 overflow-y-auto whitespace-pre leading-relaxed custom-scrollbar selection:bg-pink-500/20">
                {currentRecipe.payload}
              </pre>
            </div>
          </Card>

          {/* Tutorial Card */}
          <Card className="shadow-md border border-border/60">
            <CardTitle className="text-xs font-bold uppercase tracking-wider text-muted mb-3 flex items-center gap-1">
              <BookOpen className="w-4 h-4 text-indigo-400" /> Educational Tutorial & Guide
            </CardTitle>
            <div className="text-xs text-fg/85 leading-relaxed space-y-4 prose prose-invert max-w-none prose-sm select-text">
              {/* Manual parsing of simplistic markdown headers inside tutorial */}
              {currentRecipe.tutorial.split("\n\n").map((para, i) => {
                if (para.startsWith("### ")) {
                  return (
                    <h3 key={i} className="text-sm font-bold text-fg/90 border-b border-border/20 pb-1.5 mt-4">
                      {para.replace("### ", "")}
                    </h3>
                  );
                } else if (para.startsWith("#### ")) {
                  return (
                    <h4 key={i} className="text-xs font-bold text-accent mt-2">
                      {para.replace("#### ", "")}
                    </h4>
                  );
                } else if (para.startsWith("- ") || para.startsWith("1. ")) {
                  return (
                    <ul key={i} className="list-disc pl-5 space-y-1 my-2">
                      {para.split("\n").map((li, j) => (
                        <li key={j} className="text-fg/80">
                          {li.replace(/^(-|\d\.)\s+/, "")}
                        </li>
                      ))}
                    </ul>
                  );
                }
                return (
                  <p key={i} className="text-fg/80 my-2">
                    {para}
                  </p>
                );
              })}
            </div>
          </Card>
        </div>
      </div>

      {/* Interactive SVG Learning Rate Scheduler Simulator Card */}
      <Card className="mt-6 shadow-lg border border-border/60 overflow-hidden relative">
        <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
          <Sliders className="w-32 h-32 text-purple-400" />
        </div>
        <CardTitle className="flex items-center gap-2 mb-1 text-purple-400">
          <Sliders className="w-4 h-4 text-purple-400" />
          Interactive Learning Rate Scheduler Simulator
        </CardTitle>
        <p className="text-xs text-muted mb-4">
          Adjust hyperparameter decay variables below to visually simulate learning rate steps profiles before launching training runs.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
          {/* Controls Sliders */}
          <div className="space-y-4 text-xs lg:border-r lg:border-border/30 lg:pr-6">
            <div>
              <div className="flex justify-between font-semibold mb-1">
                <span className="text-muted">Peak Learning Rate:</span>
                <span className="font-mono text-purple-300">{peakLr.toExponential(2)}</span>
              </div>
              <input
                type="range"
                min="1e-5"
                max="1e-3"
                step="1e-5"
                className="w-full h-1.5 bg-border/40 rounded-lg appearance-none cursor-pointer accent-purple-500"
                value={peakLr}
                onChange={(e) => setPeakLr(parseFloat(e.target.value))}
              />
            </div>

            <div>
              <div className="flex justify-between font-semibold mb-1">
                <span className="text-muted">Warmup Steps:</span>
                <span className="font-mono text-purple-300">{warmupSteps} steps</span>
              </div>
              <input
                type="range"
                min="0"
                max="3000"
                step="100"
                className="w-full h-1.5 bg-border/40 rounded-lg appearance-none cursor-pointer accent-purple-500"
                value={warmupSteps}
                onChange={(e) => setWarmupSteps(parseInt(e.target.value))}
              />
            </div>

            <div>
              <div className="flex justify-between font-semibold mb-1">
                <span className="text-muted">Total Steps:</span>
                <span className="font-mono text-purple-300">{totalSteps} steps</span>
              </div>
              <input
                type="range"
                min="4000"
                max="20000"
                step="500"
                className="w-full h-1.5 bg-border/40 rounded-lg appearance-none cursor-pointer accent-purple-500"
                value={totalSteps}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  setTotalSteps(val);
                  if (warmupSteps > val * 0.4) {
                    setWarmupSteps(Math.floor(val * 0.2));
                  }
                }}
              />
            </div>

            <div>
              <div className="flex justify-between font-semibold mb-1">
                <span className="text-muted">Cosine Decay Cycles:</span>
                <span className="font-mono text-purple-300">{cosineCycles} cycles</span>
              </div>
              <input
                type="range"
                min="1"
                max="3"
                step="1"
                className="w-full h-1.5 bg-border/40 rounded-lg appearance-none cursor-pointer accent-purple-500"
                value={cosineCycles}
                onChange={(e) => setCosineCycles(parseInt(e.target.value))}
              />
            </div>
          </div>

          {/* SVG Graph Plotter */}
          <div className="lg:col-span-2 flex flex-col items-center">
            <div className="w-full bg-black/60 border border-border/40 rounded-xl p-4 shadow-inner relative overflow-hidden">
              <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-40 overflow-visible">
                <defs>
                  <linearGradient id="area-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(168, 85, 247, 0.25)" />
                    <stop offset="100%" stopColor="rgba(168, 85, 247, 0)" />
                  </linearGradient>
                </defs>

                {/* Gridlines */}
                <line x1="0" y1={height - 15} x2={width} y2={height - 15} stroke="rgba(255,255,255,0.07)" strokeWidth="1" />
                <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="rgba(255,255,255,0.07)" strokeWidth="1" />
                <line x1="0" y1="15" x2={width} y2="15" stroke="rgba(255,255,255,0.07)" strokeWidth="1" />
                
                {/* Y-axis Labels */}
                <text x="5" y="22" fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="monospace">Peak LR</text>
                <text x="5" y={height / 2 + 3} fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="monospace">0.5x LR</text>
                <text x="5" y={height - 20} fill="rgba(255,255,255,0.4)" fontSize="8" fontFamily="monospace">0</text>

                {/* Warmup vertical line indicator */}
                {warmupSteps > 0 && (
                  <line
                    x1={(warmupSteps / totalSteps) * width}
                    y1="5"
                    x2={(warmupSteps / totalSteps) * width}
                    y2={height - 15}
                    stroke="rgba(168, 85, 247, 0.2)"
                    strokeDasharray="4 4"
                    strokeWidth="1.5"
                  />
                )}

                {/* Shaded Area Under the Path */}
                <path d={areaPathData} fill="url(#area-grad)" />

                {/* Glowing neon purple path */}
                <path d={pathData} fill="none" stroke="rgb(168, 85, 247)" strokeWidth="3.5" strokeLinecap="round" className="drop-shadow-[0_0_8px_rgba(168,85,247,0.7)]" />
              </svg>

              <div className="flex justify-between text-[9px] font-mono text-muted mt-2 border-t border-border/20 pt-2 px-1">
                <span>Step 0 (Start)</span>
                {warmupSteps > 0 && (
                  <span className="text-purple-300 font-semibold">Warmup: {warmupSteps} steps</span>
                )}
                <span>Step {totalSteps} (End)</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.2);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(168, 85, 247, 0.3);
          border-radius: 2px;
        }
      `}</style>
    </>
  );
}
