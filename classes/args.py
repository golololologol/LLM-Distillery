import argparse
import tomllib
import os
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class SegmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handling: Literal["active", "context", "disabled"]


# --- v8 specials helpers -------------------------------------------------

# Resolution paths for ``inherit_specials`` lookups (relative to repo root).
_SPECIALS_DIR_RELATIVE = os.path.join("example_configs", "specials")


def _load_specials_library_entry(name: str) -> dict[str, list[str]]:
    """Resolve ``inherit_specials = "<name>"`` to a ``[specials]`` dict.

    Looks for ``example_configs/specials/<name>.toml`` (TOML file with a top
    level ``[specials]`` table or with the slot keys at top level).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, _SPECIALS_DIR_RELATIVE, f"{name}.toml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"inherit_specials={name!r}: no file at {path}"
        )
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    if "specials" in raw and isinstance(raw["specials"], dict):
        return _coerce_specials_dict(raw["specials"])
    return _coerce_specials_dict(raw)


def _coerce_specials_dict(raw: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, (list, tuple)):
            out[k] = [str(s) for s in v if s is not None]
        else:
            raise ValueError(
                f"specials slot {k!r}: expected string or list of strings, got {type(v).__name__}"
            )
    return out


def _merged_specials(
    inherit: Optional[str],
    overrides: Optional[Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    base: dict[str, list[str]] = {}
    if inherit:
        base.update(_load_specials_library_entry(inherit))
    for k, v in (overrides or {}).items():
        base[k] = list(v)
    return base


def str2bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif isinstance(value, str) and value.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class FormattedModelConfig(BaseModel):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    chat_template: Optional[str] = Field(None, description="Inline Jinja2 chat template override.")
    chat_template_path: Optional[str] = Field(None, description="Path to a file containing the Jinja2 chat template.")
    render_options: Dict[str, Any] = Field(default_factory=dict, description="Pass-through kwargs for apply_chat_template.")
    tokenizer_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Extra kwargs forwarded to AutoTokenizer.from_pretrained().")
    supports_reasoning: bool = Field(True)
    supports_tool_calls: bool = Field(True)
    segments: Dict[str, SegmentConfig] = Field(default_factory=dict)
    inherit_specials: Optional[str] = Field(None, description="Name of a specials TOML in example_configs/specials/ to inherit from.")
    specials: Dict[str, List[str]] = Field(default_factory=dict, description="Per-slot override of the specials map.")

    @field_validator('specials', mode='before')
    @classmethod
    def coerce_specials(cls, v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return _coerce_specials_dict(v)
        return v

    @field_validator('segments', mode='before')
    @classmethod
    def coerce_segments(cls, v):
        if isinstance(v, dict):
            return {seg_type: SegmentConfig(**seg_config) if isinstance(seg_config, dict) else seg_config for seg_type, seg_config in v.items()}
        return v

    @model_validator(mode='after')
    def validate_segments(self):
        supported = {"content"}
        if self.supports_reasoning:
            supported.add("reasoning")
        if self.supports_tool_calls:
            supported.update({"tool_call", "tool_name", "tool_arguments"})
        owner = type(self).__name__.replace("Config", "").lower()
        for k, seg in self.segments.items():
            if k not in supported:
                raise ValueError(
                    f"{owner} does not support segment {k!r} (supports_reasoning={self.supports_reasoning}, supports_tool_calls={self.supports_tool_calls}); remove this entry"
                )
            if seg.handling not in ("active", "context", "disabled"):
                raise ValueError(f"{owner.capitalize()} segment '{k}' handling must be 'active', 'context', or 'disabled', got '{seg.handling}'")
            if k == "content" and seg.handling == "disabled":
                raise ValueError(f"{owner.capitalize()} segment 'content' cannot be disabled")
        return self

    def effective_specials(self) -> Dict[str, List[str]]:
        """Resolved specials map (after ``inherit_specials`` + per-key overrides)."""
        return _merged_specials(self.inherit_specials, self.specials)

    def resolve_chat_template(self) -> Optional[str]:
        if self.chat_template_path:
            with open(self.chat_template_path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Support both raw Jinja files and {"chat_template": "..."} JSON
            # (the format used by Mistral's chat_template.json).
            if self.chat_template_path.endswith(".json"):
                import json as _json
                parsed = _json.loads(raw)
                if isinstance(parsed, dict) and "chat_template" in parsed:
                    return parsed["chat_template"]
            return raw
        return self.chat_template

    def effective_segments(self) -> dict[str, str]:
        result = {"content": "active"}
        for k, seg in self.segments.items():
            result[k] = seg.handling
        return result


class TeacherConfig(FormattedModelConfig):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    model_path: Optional[str] = Field(None, min_length=1)
    context_len: Optional[int] = Field(None, gt=0)
    backend_type: Optional[str] = Field(None)
    backend_params: Dict[str, Any] = Field(default_factory=dict)
    temperature: float = Field(1.0, gt=0)
    merge_weight: float = Field(1.0, gt=0)

    @property
    def can_collect(self) -> bool:
        return self.backend_type is not None

    @model_validator(mode='after')
    def validate_collectable(self):
        if self.can_collect and not self.model_path:
            raise ValueError("model_path is required for teacher configs with a backend section.")
        if self.can_collect and self.context_len is None:
            raise ValueError("context_len is required for teacher configs with a backend section.")
        return self


class StudentConfig(FormattedModelConfig):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    model_path: str = Field(..., min_length=1)
    context_len: int = Field(..., gt=0)
    freeze_layers: List[str] = Field(...)
    attn_implementation: str = Field("eager")
    save_final_training_state: bool = Field(...)
    save_training_state_every_n_epochs: Optional[float] = Field(None, gt=0)
    resume_from: Optional[str] = Field(None)
    training_temperature: float = Field(1.0, gt=0)
    event_remap: Dict[str, str] = Field(default_factory=dict, description="{slot_name: 'drop' | other_slot_name} per-student event-channel remap.")


class CanonicalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    on_multi_think: Literal["error", "first_only", "warn_first_only"] = "error"
    inline_reasoning_mode: Literal["disabled", "think_tags"] = "disabled"


class PipelineConfig(BaseModel):
    """
    Configuration class for the pipeline, containing all necessary parameters.\\
    This class uses Pydantic for validation and serialization.
    
    If you want to add a new parameter, please follow the guidelines below:
    1. Choose the correct section for the parameter based on its purpose (e.g., Paths, Cache settings, etc.).
    2. Choose a descriptive name for the parameter that reflects its purpose.
    3. Add the parameter to the appropriate section in this class and to the `config.toml` file.
    4. Make sure to add a description for the parameter in the pydantic `Field` and the docstring.
    5. If you think that the parameter must be validated, add the appropriate validators to the parameter.

    Example parameter:
    ```
    new_parameter: type_of_param = Field(..., description="User-facing description of the new parameter. {type_of_param}")
    \"\"\"Developer-facing description of the new parameter.\"\"\"
    ```
    
    Pydantic Field validators:
    - `Field(..., gt=x)` gt - greater than
    - `Field(..., ge=x)` ge - greater than or equal to
    - `Field(..., lt=x)` lt - less than
    - `Field(..., le=x)` le - less than or equal to
    - ...
    """
    model_config = ConfigDict(extra="forbid")
    
    
    # Paths
    cache_folder: str = Field(..., min_length=1, description="Directory for cache storage. Ideally should be an empty folder. {string}")
    """Directory for cache storage."""
    
    dataset_path: str = Field(..., min_length=1, description="Path to the training dataset. {string}")
    """Path to the training dataset."""
    
    validation_dataset_path: str = Field(..., min_length=1, description="Path to the validation dataset. {string}")
    """Path to the validation dataset."""
    
    teacher_configs_path: str = Field(..., min_length=1, description="Directory containing teacher model configurations, or a path to just one teacher config directly. {string}")
    """Directory containing teacher model configurations, or a path to just one teacher config directly."""
    
    student_config_path: str = Field(..., min_length=1, description="Path to the student model configuration file. {string}")
    """Path to the student model configuration file."""


    # Dataset format
    dataset_format: str = Field("auto", description="Named strategy for converting raw dataset rows to canonical messages. 'auto' detects from row keys (openai/sharegpt/alpaca/completion). See classes/dataset_formats.py for the full registry. {string}")
    """Named dataset-format strategy. See classes/dataset_formats.py."""


    # General model settings
    save_roles: List[str] = Field(default_factory=lambda: ["assistant"], description="List of roles to save into the hdf5 dataset. If empty, all roles will be saved. {string list}")
    """List of roles to save into the hdf5 dataset. If empty, all roles will be saved."""
    
    auto_approve: bool = Field(False, description="Skip interactive confirmation prompts for dataset modifications (deletions, renames). Warnings are still printed. {bool}")
    """When True, dataset sync operations proceed without waiting for user input. Warnings are still printed."""

    canonicalization: CanonicalizationConfig = Field(default_factory=CanonicalizationConfig, description="Canonicalization options applied at dataset load.")
    """Canonicalization options applied at dataset load."""


    # Collection settings
    marg_chunk_inference: int = Field(256, gt=0, description="Token chunk size for byte marginalization during inference. Controls peak memory usage. {int}")
    """Token chunk size for byte marginalization during inference."""

    marg_chunk_training: int = Field(256, gt=0, description="Token chunk size for byte marginalization during training. Lower values reduce memory at slight speed cost. {int}")
    """Token chunk size for byte marginalization during training."""


    # Training settings
    num_epochs: int = Field(..., gt=0, description="Number of training epochs. {int}")
    """Number of training epochs."""
    
    num_warmup_steps: int = Field(0, ge=0, description="Number of warmup steps for learning rate. {int}")
    """Number of warmup steps for learning rate."""
    
    batch_size: int = Field(..., gt=0, description="Training batch size. {int}")
    """Batch size for training."""
    
    grad_accum_batches: int = Field(1, gt=0, description="Number of gradient accumulations before calling optimizer.step(). {int}")
    """Number of gradient accumulations before optimizer step."""
    
    grad_checkpointing: bool = Field(True, description="Enable gradient checkpointing for memory savings. {bool}")
    """Flag to enable gradient checkpointing."""
    
    torch_compile: bool = Field(False, description="Enable torch.compile on student model. {bool}")
    """JIT-compile the student model for faster training."""
    
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = Field("default", description="torch.compile mode. {str}")
    """Mode for torch.compile during training."""
    
    torch_compile_backend: Literal["inductor", "cudagraphs", "aot_eager", "eager"] = Field("inductor", description="torch.compile backend. {str}")
    """Backend for torch.compile during training."""
    
    liger_kernel: bool = Field(False, description="Enable Liger Kernel fused Triton ops (RMSNorm, RoPE, SwiGLU). Requires liger-kernel package. {bool}")
    """Flag to enable Liger Kernel fused Triton ops (RMSNorm, RoPE, SwiGLU). Requires liger-kernel package and triton."""
    
    lr: float = Field(..., gt=0, description="Learning rate. {float}")
    """Learning rate."""
    
    adam_betas: Tuple[float, float] = Field((0.9, 0.999), description="Betas for Adam-like optimizers. Must be a list of two floats. {list}")
    """Betas for Adam-like optimizers."""
    
    adam_decay: float = Field(0.01, ge=0, description="Decay for Adam-like optimizers. {float}")
    """Decay for Adam-like optimizers."""
    
    lr_decay_start: float = Field(0.9, ge=0, le=1, description="Start decaying learning rate to 0 at this percentage of total training steps (0.1 for 10% of total training steps). {float}")
    """Start ratio for lr decay."""
    
    alpha: float = Field(0.5, description="Weighting factor for weighted losses. {float}")
    """Weighting factor for weighted losses."""
    
    loss_type: Literal["abomination", "skew_kl", "akl", "wasserstein", "jsd", "hellinger", "forward_kl", "reverse_kl"] = Field("abomination", description="Loss function type. {string}")
    """Loss function type."""

    entropy_weighting: bool = Field(False, description="Scale per-position loss by teacher distribution entropy. Downweights sparse tokenizer-artifact positions. {bool}")
    """Flag to enable entropy weighting of the loss."""

    # v8 event channel (plan §9, §12)
    lambda_event: float = Field(0.0, ge=0, description="Global weight for the event-channel loss. 0 disables the event arm entirely. {float}")
    """Global weight for the event-channel loss (plan v8)."""

    event_loss: Literal["jsd", "kl_fwd", "kl_rev", "hellinger"] = Field("jsd", description="Divergence function applied on the |E|-event simplex. {string}")
    """Divergence function applied on the event simplex."""

    event_extensions: List[str] = Field(default_factory=list, description="Named event-vocab extensions appended after the 15-slot core. Empty = core-only. {list[string]}")
    """Named event-vocab extensions (plan v8 §2)."""

    lr_scheduler: str = Field("wsd", description="Learning rate scheduler name. {string}")
    """Name of the learning rate scheduler to use."""
    
    optimizer: Literal["adamw_torch", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit", "sgd", "rmsprop", "rmsprop8bit", "rmsprop32bit", "adagrad", "apollo", "apollomini", "schedulefree", "muon"] = Field(..., description="Optimizer name. {string}")
    """Name of the optimizer to use."""
    
    max_grad_norm: float = Field(1.0, ge=0, description="Maximum gradient norm for clipping. Set to 0 to disable. {float}")
    """Maximum gradient norm for clipping."""

    seed: int = Field(42, description="Random seed for reproducible data shuffling. {int}")
    """Random seed for reproducible data shuffling."""

    data_order: Literal["shuffle", "random", "sorted"] = Field("random", description="Order of samples during training. {string}")
    """Name of the order of samples to use during training."""
    
    training_precision: Literal["fp16", "fp32", "bf16", "4bit", "8bit"] = Field("bf16", description="Training precision. {string}")
    """Name of the precision to use for training."""
    
    train_on: str | list[str] = Field("all", description="Which teacher(s) to train on. 'all' = use all available teachers (single teacher trains directly, multiple merges with per-teacher weights). Can be a specific teacher name, or a list of teacher names to merge a subset. {string | list[string]}")
    """Which teacher(s) to train on."""
    
    training_strategy: Literal["ddp", "fsdp2", "naive", "naive_layer_split"] = Field("naive", description="Training parallelism strategy. Options: 'ddp', 'fsdp2', 'naive'/'naive_layer_split'. {string}")
    """Training parallelism strategy."""


    # Validation & Saving
    validate_every_n_epochs: float = Field(1.0, gt=0, description="Validation frequency measured in epochs. Accepts floating point values. {float}")
    """Validation frequency in epochs."""
    
    best_metric: Literal["train_loss", "ce", "kl"] = Field("train_loss", description="Metric used to determine best validation checkpoint. Options: 'train_loss', 'ce', 'kl'. {string}")
    """Which validation metric to track for 'best' model selection."""
    
    save_student_every_n_epochs: float = Field(1.0, gt=0, description="Frequency of saving student model in epochs. {float}")
    """Frequency to save the student model in epochs."""
    
    save_best_model: bool = Field(True, description="Save the student model when a new best validation metric is reached. {bool}")
    """Save model checkpoint on new best validation."""
    
    save_best_state: bool = Field(False, description="Save training state when a new best validation metric is reached. {bool}")
    """Save training state on new best validation."""
    
    keep_last_n_checkpoints: Optional[int] = Field(None, ge=1, description="Number of most recent checkpoints to keep. Older checkpoints will be deleted. Set to None to keep all checkpoints. {int}")
    """Number of most recent checkpoints to keep."""

    # Multi-GPU / device settings
    num_gpu0_layers: Optional[int] = Field(None, ge=0, description="Number of layers for GPU 0. Required when device_map = \"custom\". {int}")
    """Number of layers on GPU 0."""
    
    device_map: str = Field("auto", description="Device mapping strategy. {string}")
    """Name of the device mapping strategy to use."""
    
    max_memory: Dict[str, str] = Field(default_factory=dict, description="Maximum memory allocation for each device. Keys: GPU indices or 'cpu'. {dict[str, str]}")
    """Maximum memory allocation for each device."""
    
    @property
    def max_memory_hf(self) -> dict:
        """max_memory with GPU indices as int, for HuggingFace from_pretrained."""
        return {(k if k == "cpu" else int(k)): v for k, v in self.max_memory.items()}
    
    multi_gpu: bool = Field(False, description="Whether to do multi-GPU training. {bool}")
    """Flag to enable multi-GPU training."""
    
    wandb_comment: str = Field("", description="A comment for Weights and Biases logging for the given training run. {string}")
    """Comment for Weights and Biases logging for the given training run."""
    
    wandb_project: str = Field("LLM Distillation", description="Weights and Biases project name. {string}")
    """Weights and Biases project name."""


    @field_validator('training_strategy', mode='before')
    def normalize_training_strategy(cls, v):
        if v == "naive":
            return "naive_layer_split"
        return v

    @field_validator('dataset_format')
    def validate_dataset_format(cls, v):
        from classes.dataset_formats import REGISTRY
        if v != "auto" and v not in REGISTRY:
            raise ValueError(f"Unknown dataset_format {v!r}. Available: {sorted(REGISTRY)} or 'auto'.")
        return v

    @field_validator('train_on', mode='before')
    def validate_train_on(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("train_on list must not be empty")
        return v

    @field_validator('adam_betas', 'max_memory', mode='before')
    def convert_fields(cls, v, info):
        if info.field_name == 'adam_betas' and isinstance(v, list):
            return tuple(v)
        if info.field_name == 'max_memory' and isinstance(v, dict):
            result = {}
            for k, val in v.items():
                key = "cpu" if str(k).lower() == "cpu" else str(int(k))
                result[key] = val
            return result
        return v

    @field_validator('dataset_path', 'validation_dataset_path')
    def validate_dataset_paths(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"Dataset file not found: {v}")
        return v

    @model_validator(mode='after')
    def validate_incompatible_options(self):
        if self.device_map == "custom" and self.num_gpu0_layers is None:
            raise ValueError("num_gpu0_layers is required when device_map = 'custom'.")
        if self.torch_compile and self.liger_kernel:
            raise ValueError(
                "torch_compile and liger_kernel cannot both be enabled. "
                "Liger Kernel uses torch.autograd.Function which causes graph breaks under torch.compile, "
                "resulting in severe compilation overhead. Disable one of them."
            )
        return self


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}!\nPlease ensure the file exists in the specified location."
        )
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def generate_cli_arguments(parser: argparse.ArgumentParser, model: type[BaseModel]):
    """
    Auto-generate CLI arguments from the Pydantic model's fields.
    """
    for field_name, model_field in model.model_fields.items():
        arg_name = f"--{field_name}"
        field_type = model_field.annotation
        help_text = model_field.description
        kwargs = {"help": help_text, "dest": field_name, "required": False}
        origin = get_origin(field_type)

        # Unwrap Optional[X] → X
        if origin is Union:
            inner = [a for a in get_args(field_type) if a is not type(None)]
            if inner:
                field_type = inner[0]
                origin = get_origin(field_type)

        # Handle Literal types → str with choices
        if origin is Literal:
            kwargs["type"] = str
            kwargs["choices"] = list(get_args(field_type))
        elif origin in (list, tuple):
            kwargs["nargs"] = "+"
            args_type = get_args(field_type)
            kwargs["type"] = args_type[0] if args_type else str
        elif field_type == bool:
            kwargs["type"] = str2bool
        elif field_type in (str, int, float):
            kwargs["type"] = field_type
        else:
            kwargs["type"] = str

        parser.add_argument(arg_name, **kwargs)


def merge_config(cli_args: dict, config: dict) -> dict:
    """
    Merge CLI arguments with config values. CLI args override config.
    """
    merged = config.copy()
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged


def _suggest_typos(error: ValidationError, valid_fields: set[str]) -> str:
    import difflib
    messages = []
    for e in error.errors():
        if e['type'] == 'extra_forbidden':
            field = e['loc'][0]
            close = difflib.get_close_matches(str(field), valid_fields, n=3, cutoff=0.6)
            if close:
                messages.append(f"  Unknown field '{field}' - did you mean: {', '.join(close)}?")
            else:
                messages.append(f"  Unknown field '{field}' - not a valid config parameter.")
    return '\n'.join(messages) if messages else ''


def get_config(config_path=None) -> tuple[PipelineConfig, bool, bool, bool]:
    """
    Loads TOML configuration and merges it with CLI arguments.
    Returns (config, validate_only, collect_only, train_only).
    """
    default_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.toml')
    
    parser = argparse.ArgumentParser(description="LLM-Distillery: Teacher-student knowledge distillation pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to the pipeline config TOML file. Defaults to config.toml in the project root.")
    parser.add_argument("--validate", action="store_true", default=False, help="Validate configuration and exit without running the pipeline.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--collect-only", action="store_true", default=False, help="Run collection phase only, skip training.")
    mode_group.add_argument("--train-only", action="store_true", default=False, help="Run training phase only using pre-collected data. Teacher configs not required.")
    generate_cli_arguments(parser, PipelineConfig)
    args = parser.parse_args()
    cli_args = vars(args)
    
    config_path = cli_args.pop("config") or config_path or default_config_path
    validate_only = cli_args.pop("validate")
    collect_only = cli_args.pop("collect_only")
    train_only = cli_args.pop("train_only")
    toml_config = load_config(config_path)

    missing_keys = [field for field, info in PipelineConfig.model_fields.items() if info.is_required() and field not in toml_config]
    if missing_keys:
        print(f"Error: config.toml is missing the following parameters: {', '.join(missing_keys)}")
        print("\nPlease add them to the config file!")
        sys.exit(1)

    merged_params = merge_config(cli_args, toml_config)
    try:
        return PipelineConfig(**merged_params), validate_only, collect_only, train_only
    except ValidationError as e:
        suggestions = _suggest_typos(e, set(PipelineConfig.model_fields))
        if suggestions:
            print(f"\nConfig errors:\n{suggestions}")
            sys.exit(1)
        raise


_NON_BACKEND_KEYS = {"segments", "tokenizer_kwargs"}


def _parse_teacher_toml(raw: dict, filename: str) -> TeacherConfig:
    backend_type = None
    backend_params = {}
    core_params = {}
    for k, v in raw.items():
        if isinstance(v, dict) and k not in _NON_BACKEND_KEYS:
            if backend_type is not None:
                raise ValueError(f"Teacher config {filename} has multiple backend sections: {backend_type}, {k}")
            backend_type = k
            backend_params = v
        else:
            core_params[k] = v
    if backend_type is not None:
        core_params['backend_type'] = backend_type
        core_params['backend_params'] = backend_params
    return TeacherConfig(**core_params)


def load_teacher_configs(configs_path: str) -> list[tuple[str, TeacherConfig]]:
    if os.path.isfile(configs_path):
        name = os.path.basename(configs_path).removesuffix('.toml')
        return [(name, _parse_teacher_toml(load_config(configs_path), configs_path))]

    if not os.path.isdir(configs_path):
        raise FileNotFoundError(f"Teacher configs path not found: {configs_path}")

    configs = []
    for filename in sorted(os.listdir(configs_path)):
        if filename.endswith('.toml'):
            filepath = os.path.join(configs_path, filename)
            teacher_name = filename.removesuffix('.toml')
            configs.append((teacher_name, _parse_teacher_toml(load_config(filepath), filename)))

    if not configs:
        raise FileNotFoundError(f"No TOML config files found in: {configs_path}")

    return configs


def load_student_config(config_path: str) -> StudentConfig:
    """
    Loads a single student configuration from a TOML file.
    """
    if os.path.isdir(config_path):
        toml_files = [f for f in os.listdir(config_path) if f.endswith('.toml')]
        if len(toml_files) == 0:
            raise FileNotFoundError(f"No TOML config files found in: {config_path}")
        if len(toml_files) > 1:
            raise ValueError(f"Multiple TOML config files found in {config_path}: {toml_files}\nStudent config path must point to a single config file, or a directory containing exactly one.")
        config_path = os.path.join(config_path, toml_files[0])

    raw = load_config(config_path)
    return StudentConfig(**raw)


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.toml')
    config, *_ = get_config(config_path=config_path)
    print(config.model_dump_json(indent=4))
