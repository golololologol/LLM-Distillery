# LLM-Distillery

Offline white-box knowledge distillation pipeline for LLMs. Collects teacher logit distributions into HDF5, then trains a student model from the stored distributions — completely decoupling collection from training so datasets are reusable, shareable, and incrementally updateable.

## Features

- **Offline distillation** — Collection and training are separate phases. Collected HDF5 datasets are reusable and shareable across runs and student models.
- **Byte-level distributions** — Stores complete byte-level marginals via CUDA trie marginalization. Enables Any-to-Any tokenizer distillation: the teacher and student can have completely different vocabularies.
- **Single and multi-teacher** — Merge distributions from multiple teachers with configurable per-teacher weights. Multi-teacher merging is automatic when `train_on` lists more than one teacher.
- **Quantized teacher inference** — ExLlamaV2 and ExLlamaV3 backends for efficient inference from EXL2/GGUF quantized models. Additional backends are easy to plug in via the registry in `classes/inference/`.
- **Event vocabulary** — 15-slot frozen event ontology tracks model-output control events (turn boundaries, reasoning delimiters, tool-call boundaries, etc.) across all tokenizers. Stored alongside byte distributions in HDF5 and used to align cross-tokenizer training.
- **Typed segment handling** — Conversations are split into typed segments (`content`, `reasoning`, `tool_call`). Each segment type can be independently set to `active` (loss-bearing), `context` (rendered but no loss), or `disabled` (stripped entirely), per model.
- **Resumable collection** — SHA-based dataset sync. Interrupt and resume mid-collection. Add new rows to the source JSONL and only the new samples are collected.
- **Custom CUDA kernels** — JIT-compiled fused kernels for byte marginalization, inference, and training. Compiled once on first use and cached.
- **Rich training stack** — Gradient checkpointing, `torch.compile`, Liger Kernel fused Triton ops (RMSNorm, RoPE, SwiGLU), multi-GPU (DDP / FSDP2 / naive layer split), WandB logging, checkpoint rotation, training state save/resume.
- **Broad optimizer support** — AdamW variants, Apollo/ApolloMini, ScheduleFree, Muon, SGD, RMSProp, Adagrad, and more.
- **Multiple loss functions** — `abomination` (composite), `skew_kl`, `akl`, `wasserstein`, `jsd`, `hellinger`, `forward_kl`, `reverse_kl`. Entropy weighting available to downweight sparse positions from tokenizer boundary mismatches.
- **Agentic dataset support** — Built-in format strategies for Hermes inline tags, per-block-role datasets (jupyter\_agent, ToolACE, NemotronChat), and Nemotron post-training format. Plus a one-function API for adding custom formats.
- **Windows and Linux support** — Including multi-GPU on Linux/WSL.

TinyLlama 1.1B full-parameter distillation in ~8 GB of VRAM, at ~3 500 tokens/s training and ~18 000 tokens/s validation on a single RTX 3090.

<img width="3341" height="1692" alt="image" src="https://github.com/user-attachments/assets/fc3817ad-eb73-43ef-9bc8-5ac2c7f7df7f" />


## Quickstart

### 1. Install

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux:   source .venv/bin/activate

pip install -r requirements.txt
```

PyTorch must be installed separately with CUDA support. See [pytorch.org](https://pytorch.org/get-started/locally/).

> **Windows multi-GPU note:** DDP and FSDP2 require the Gloo backend, which is broken in PyTorch ≥ 2.8 on Windows ([pytorch/pytorch#150381](https://github.com/pytorch/pytorch/issues/150381)). Use `training_strategy = "naive_layer_split"` for multi-GPU on Windows, or downgrade to PyTorch 2.7.0. Linux/WSL is unaffected.

Optional packages:
- `liger-kernel` + `triton` — fused Triton ops; set `liger_kernel = true` in pipeline config. Cannot be combined with `torch_compile`.
- `flash-attn` — Flash Attention 2; set `attn_implementation = "flash_attention_2"` in student config.
- `huggingface-hub` — auto-downloads models from HuggingFace (bundled with `transformers`).

### 2. Configure

Three config files control the pipeline. Fully-annotated templates live in [`example_configs/`](example_configs/).

---

**`config.toml`** — Main pipeline config (paths, collection settings, training hyperparameters, optimizer, loss, multi-GPU, WandB).

Key fields:
```toml
cache_folder   = "data/my_run/teacher_cache"
dataset_path   = "data/train.jsonl"
validation_dataset_path = "data/val.jsonl"
teacher_configs_path = "teacher_configs/"
student_config_path  = "student_configs/qwen_0.6b.toml"

loss_type   = "abomination"   # abomination | skew_kl | akl | wasserstein | jsd | hellinger | forward_kl | reverse_kl
optimizer   = "apollomini"    # apollomini | apollo | adamw | muon | schedulefree | ...
lr_scheduler = "wsd"
train_on    = "all"           # "all" | "teacher_name" | ["teacher1", "teacher2"]
training_strategy = "ddp"     # ddp | fsdp2 | naive_layer_split | naive
```

---

**`teacher_configs/*.toml`** — One file per teacher. The filename (without `.toml`) becomes the teacher name used throughout the pipeline.

```toml
model_path  = "Qwen/Qwen3-8B"          # HuggingFace ID, local path, or "repo:revision"
context_len = 2048
temperature = 1.0                        # higher = softer distribution
merge_weight = 1.0                       # relative weight when merging teachers

supports_reasoning  = true
supports_tool_calls = true

[exllamav2]                              # use [exllamav3] for the V3 backend
reserve_vram         = [2.5]            # GB reserved per GPU for KV cache / activations
batch_size           = 4
seq_chunk_len        = 2048
num_inference_workers = 1
one_worker_per_gpu   = false
```

A teacher config with no backend section (`[exllamav2]` / `[exllamav3]`) is a **metadata-only** config — it references a previously collected HDF5 and contributes only its `merge_weight` during training. Useful for mixing teachers collected on different machines.

---

**`student_configs/*.toml`** — Student model config.

```toml
model_path  = "Qwen/Qwen3-0.6B"
context_len = 2048
attn_implementation = "eager"           # eager | sdpa | flash_attention_2
freeze_layers = []                       # list of parameter name patterns to freeze
save_final_training_state = false
training_temperature = 1.0

# Optional: resume from a saved training state
# resume_from = "cache/states/epoch_2"

# [segments.reasoning]
# handling = "active"    # active | context | disabled

# [segments.tool_call]
# handling = "active"
```

### 3. Specials Maps

The pipeline aligns special tokens (turn delimiters, reasoning tags, tool-call markers) across teacher and student tokenizers using a **specials map** — a `{slot_name: ["<token>", ...]}` table in each model config.

Pre-built maps for common model families live in [`example_configs/specials/`](example_configs/specials/) and can be inherited with a single line:

```toml
inherit_specials = "qwen_chatml"   # loads example_configs/specials/qwen_chatml.toml
```

Available presets: `chatglm4`, `cohere_command_r`, `command_r`, `command_r_plus`, `deepseek_v3`, `exaone`, `gemma`, `gemma4`, `granite`, `harmony`, `hermes_chatml`, `internlm2`, `llama3`, `llama4`, `mistral_small_3.1`, `mistral_v3`, `openchat`, `phi`, `qwen_chatml`, `yi`, `zephyr`.

Individual slots can be overridden on top of an inherited preset:
```toml
inherit_specials = "llama3"
[specials]
E_END_TURN = ["<|eot_id|>", "<|end_of_text|>"]
```

### 4. Prepare Data

Dataset format: JSONL with OpenAI-style messages (one JSON object per line):
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

Reasoning traces and tool calls are supported as structured fields on assistant messages:
```json
{"messages": [
  {"role": "user", "content": "What is 2+2?"},
  {"role": "assistant", "reasoning": "Simple arithmetic.", "content": "4.", "tool_calls": []}
]}
```

Many common dataset shapes are auto-converted: ShareGPT, Alpaca, Dolly, instruction/response, query/answer, Hermes inline tags, per-block-role agentic formats, and more. Set `dataset_format` in `config.toml`, or leave it at `"auto"` for automatic detection.

See [DATASET_FORMAT.md](DATASET_FORMAT.md) for the full format spec, the complete strategy registry, and instructions for adding custom strategies. Always verify converted data with [`tools/inspect_segments.py`](tools/inspect_segments.py) before collecting.

### 5. Run

```bash
# Full pipeline (collect then train)
python collect_and_finetune.py

# Different config file
python collect_and_finetune.py --config path/to/my_config.toml

# Validate config without running anything
python collect_and_finetune.py --validate

# Override any config value from the CLI
python collect_and_finetune.py --lr 1e-5 --batch_size 8 --loss_type wasserstein
```

## Cache Structure

```
cache_folder/
├── dataset/
│   ├── teacher_name.hdf5       # Per-teacher training distributions
│   ├── another_teacher.hdf5
│   ├── _merged.hdf5            # Auto-generated when training on multiple teachers
│   └── validation/
│       ├── teacher_name.hdf5   # Per-teacher validation distributions
│       └── another_teacher.hdf5
└── student/
    ├── trained/                 # Model checkpoints (safetensors + tokenizer)
    ├── states/                  # Training state checkpoints (optimizer, scheduler)
    └── gguf/                    # GGUF exports
```

HDF5 files store per-sample byte distributions, event distributions, segment manifests, and collection metadata. They are keyed by a SHA of the source conversation content, so the sync system can detect added, modified, or removed rows and update only what changed.

## Segment Types and Handling

Every byte of training-relevant text is tagged with a **segment type**:

| Type | Source | Configurable? |
|---|---|---|
| `content` | `message.content` | Always active (cannot be disabled) |
| `reasoning` | `assistant.reasoning` | Yes — per model via `[segments.reasoning]` |
| `tool_call` | `assistant.tool_calls[*].arguments` | Yes — per model via `[segments.tool_call]` |

Each segment can be set to:
- `active` — rendered and included in loss (default for `content`).
- `context` — rendered so the chat template stays valid, but excluded from loss.
- `disabled` — stripped entirely from the rendered output (not allowed for `content`).

`save_roles` (default: `["assistant"]`) selects which message roles produce loss-bearing segments. Roles outside this list are rendered as plain context.

## Multi-GPU Training

| Strategy | Description |
|---|---|
| `ddp` | DistributedDataParallel — model replicated on each GPU. Best for models that fit on one GPU. |
| `fsdp2` | Fully Sharded Data Parallel — shards parameters, gradients, and optimizer state. Best for large models. |
| `naive_layer_split` | Splits model layers across GPUs without distributed communication. Works on Windows. |
| `naive` | Single-device (alias for no split). |

Set `multi_gpu = true` and `training_strategy` in `config.toml`. On Windows, use `naive_layer_split` (DDP/FSDP2 require Gloo which is broken on PyTorch ≥ 2.8 for Windows).

## Tools

| Script | Purpose |
|---|---|
| [`tools/inspect_segments.py`](tools/inspect_segments.py) | Visualize segment boundaries and types in a dataset. Run before collecting. |
| [`tools/dataset_converter.py`](tools/dataset_converter.py) | Convert datasets between formats. |
| [`tools/convert_parquet.py`](tools/convert_parquet.py) | Convert Parquet files to JSONL. |
| [`tools/pack_samples.py`](tools/pack_samples.py) | Pack short samples together to reduce padding waste. |

## Contributions

Big thanks to [kalomaze](https://github.com/kalomaze) for help and keeping me sane while I was building this project!\
Also, thanks to [AlpinDale](https://github.com/AlpinDale) for giving access to compute during the development!

If you want to contribute to this project, feel free!\
Open issues when you encounter them, and make PRs when you feel like it.
