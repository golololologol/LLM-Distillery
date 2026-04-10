# LLM-Distillery

Offline white-box knowledge distillation pipeline for LLMs. Collects teacher logit distributions into HDF5, then trains a student model from the stored distributions.

## Features

- **Offline distillation** - Collection and training are separate phases. Collected datasets are reusable and shareable.
- **Byte-level distributions** - Stores complete byte-level marginals via trie marginalization. Allows Any-Any tokenizer training!
- **Single and multi-teacher** - Merge distributions from multiple teachers with configurable weights.
- **Quantized teacher inference support** - Uses ExLlamaV2 for efficient teacher inference from quantized (EXL2) models.
- **Modular inference backend architecture** - More inference engines are easy to add.
- **Custom CUDA kernels** - JIT-compiled custom fused kernels for fast and memory-efficient training and inference.
- **Resumable collection** - SHA-based dataset sync. Force-quit and continue where you left off. Or update the dataset, and collect only what is new.
- **Training features** - Gradient checkpointing, torch.compile, Liger Kernel, multi-GPU (DDP/FSDP2), WandB logging, checkpoint rotation.
- **Windows and Linux support**

## Quickstart

### 1. Install

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate

pip install -r requirements.txt
```

PyTorch must be installed separately with CUDA support. See [pytorch.org](https://pytorch.org/get-started/locally/).

> **Windows multi-GPU note:** DDP and FSDP2 training strategies require the Gloo backend, which is broken in PyTorch >= 2.8 on Windows ([pytorch/pytorch#150381](https://github.com/pytorch/pytorch/issues/150381)). Use `training_strategy = "naive_layer_split"` for multi-GPU training on Windows, or downgrade to PyTorch 2.7.0. NCCL *might*  be coming to windows and would resolve all this, but its unclear if/when would it happen ([NVIDIA windows NCCL PR](https://github.com/NVIDIA/nccl/pull/1922)). Linux/WSL is unaffected.

Optional packages:
- `liger-kernel` + `triton` - for fused Triton ops (set `liger_kernel = true`)
- `huggingface-hub` - auto-downloads models from HuggingFace (installed with transformers)

### 2. Configure

Three config files control the pipeline:

**`config.toml`** - Main pipeline config (paths, training hyperparameters, loss, optimizer, etc.)

**`teacher_configs/*.toml`** - One file per teacher model:
```toml
model_path = "TheMelonGod/Qwen3-8B-exl2:8hb-6.0bpw"
context_len = 2048
temperature = 2
merge_weight = 1.0

[exllamav2]
reserve_vram = [2.5, 0.5]
batch_size = 4
seq_chunk_len = 2048
num_inference_workers = 3
one_worker_per_gpu = true
```

**`student_configs/*.toml`** - Student model config:
```toml
model_path = "Qwen/Qwen3-0.6B"
freeze_layers = []
attn_implementation = "eager"
save_final_training_state = false
context_len = 2048
```

### 3. Prepare Data

Dataset format: JSONL with OpenAI-style messages:
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

Set `dataset_path` and `validation_dataset_path` in `config.toml`.

### 4. Run

```bash
# Full pipeline (collect + train)
python collect_and_finetune.py

# With a different config file
python collect_and_finetune.py --config path/to/my_config.toml

# Validate config without running
python collect_and_finetune.py --validate

# Override any config value from CLI
python collect_and_finetune.py --lr 1e-5 --batch_size 8
```

## Cache Structure

```
cache_folder/
├── dataset/
│   ├── distributions.hdf5      # Training distributions
│   └── validation/
│       └── distributions.hdf5  # Validation distributions
└── student/
    ├── trained/                 # Model checkpoints
    ├── states/                  # Training state checkpoints
    └── gguf/                    # GGUF exports
```

## Contributions

Big thanks to [kalomaze](https://github.com/kalomaze) for help and keeping me sane while I was building this project!\
Also, thanks to [AlpinDale](https://github.com/AlpinDale) for giving access to compute during the development!

If you want to contribute to this project, feel free!\
Open issues when you encounter them, and make PRs when you feel like it.
