import argparse
import json
import os
from typing import List, Dict, Tuple, get_origin, get_args
from pydantic import BaseModel, Field, field_validator, model_validator
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif isinstance(v, str) and v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class PipelineConfig(BaseModel):
    """
    Configuration class for the pipeline, containing all necessary parameters.\\
    This class uses Pydantic for validation and serialization.
    
    If you want to add a new parameter, please follow the guidelines below:
    1. Choose the correct section for the parameter based on its purpose (e.g., Paths, Cache settings, etc.).
    2. Choose a descriptive name for the parameter that reflects its purpose.
    3. Add the parameter to the appropriate section in this class and to the `config.json` file.
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
    
    
    # Paths
    cache_folder: str = Field(..., min_length=1, description="Directory for cache storage. Ideally should be an empty folder. {string}")
    """Directory for cache storage."""
    
    dataset_path: str = Field(..., min_length=1, description="Path to the training dataset. {string}")
    """Path to the training dataset."""
    
    validation_dataset_path: str = Field(..., min_length=1, description="Path to the validation dataset. {string}")
    """Path to the validation dataset."""
    
    teacher_models_folder: str = Field(..., min_length=1, description="Directory containing teacher models, or a path to just one teacher directly. {string}")
    """Directory containing teacher models, or a path to just one teacher directly."""
    
    student_path: str = Field(..., min_length=1, description="Path to the student model. {string}")
    """Path to the student model."""


    # Cache settings
    max_cache_size_gb: float = Field(..., gt=0, description="Maximum cache size in GB. Used to keep the main h5 dataset under this limit, and use chunked collection+training when the calculated size of the collected h5 dataset is over this limit. Only tracks the main h5 dataset's size, any misc. files/states are not counted. {float}")
    """Maximum cache size in GB."""


    # Pipeline settings
    ignore_model_type: bool = Field(..., description="If True, will let completion teachers collect instruct data, and instruct teachers completion data. Use at your own discretion. {bool}")
    """Flag to ignore teacher model type checks."""
    
    rebase_dataset: bool = Field(..., description="Rebase the dataset without safety checks. Overwrites all metadata in the h5 dataset. {bool}")
    """Flag to rebase the dataset."""


    # General model settings
    use_teachers: bool = Field(..., description="Whether to use teachers for distillation. {bool}")
    """Flag to enable teacher models for distillation."""
    
    context_len: int = Field(..., gt=0, description="Context length to collect and train on. {int}")
    """Context length for training."""
    
    save_sys_range: bool = Field(..., description="Boolean flag to save specific token ranges within conversations (system role). {bool}")
    """Flag to save system token range."""
    
    save_user_range: bool = Field(..., description="Boolean flag to save specific token ranges within conversations (user role). {bool}")
    """Flag to save user token range."""
    
    save_assistant_range: bool = Field(..., description="Boolean flag to save specific token ranges within conversations (assistant role). {bool}")
    """Flag to save assistant token range."""
    
    crop_distr_to_size: int = Field(..., gt=0, description="Crop distribution size for token filtering. Must be set to the base-model's vocabulary size. {int}")
    """Size limit for distribution cropping."""
    
    enable_topK: bool = Field(..., description="Enable top-K sampling for collecting and training. {bool}")
    """Flag to enable top-K sampling."""
    
    save_topK: int = Field(..., ge=0, description="Configure top-K sampling for collecting and training. {int}")
    """Number of top-K tokens to save."""
    
    device: str = Field(..., min_length=1, description="Main device for any single-device tensor operations (e.g., cuda:0). {string}")
    """Main device for computations (e.g., cuda:0)."""


    # Collection settings
    num_inference_workers: int = Field(..., gt=0, description="Number of inference workers to use. {int}")
    """Number of inference workers to use."""
    
    reserve_vram: List[float] = Field(..., description="Amount of VRAM to reserve per GPU during collection. {float list}")
    """Amount of VRAM to reserve per GPU."""


    # Training settings
    num_epochs: int = Field(..., gt=0, description="Number of training epochs. {int}")
    """Number of training epochs."""
    
    num_warmup_steps: int = Field(..., ge=0, description="Number of warmup steps for learning rate. {int}")
    """Number of warmup steps for learning rate."""
    
    batch_size: int = Field(..., gt=0, description="Training batch size. {int}")
    """Batch size for training."""
    
    grad_accum_batches: int = Field(..., gt=0, description="Number of gradient accumulations before calling optimizer.step(). {int}")
    """Number of gradient accumulations before optimizer step."""
    
    grad_checkpointing: bool = Field(..., description="Enable gradient checkpointing for memory savings. {bool}")
    """Flag to enable gradient checkpointing."""
    
    temperature: float = Field(..., ge=0, description="Temperature for distillation. {float}")
    """Distillation temperature."""
    
    lr: float = Field(..., gt=0, description="Learning rate. {float}")
    """Learning rate."""
    
    adam_betas: Tuple[float, float] = Field(..., description="Betas for Adam-like optimizers. Must be a list of two floats. {list}")
    """Betas for Adam-like optimizers."""
    
    adam_decay: float = Field(..., ge=0, description="Decay for Adam-like optimizers. {float}")
    """Decay for Adam-like optimizers."""
    
    lr_decay_start: float = Field(..., ge=0, le=1, description="Start decaying learning rate to 0 at this percentage of total training steps (0.1 for 10% of total training steps). {float}")
    """Start ratio for lr decay."""
    
    alpha: float = Field(..., description="Weighting factor for weighted losses. {float}")
    """Weighting factor for weighted losses."""
    
    lr_scheduler: str = Field(..., description="Learning rate scheduler name. {string}")
    """Name of the learning rate scheduler to use."""
    
    optimizer: str = Field(..., description="Optimizer name. (adam, adamw, etc.) {string}")
    """Name of the optimizer to use."""
    
    data_order: str = Field(..., description="Order of samples during training. {string}")
    """Name of the order of samples to use during training."""
    
    training_precision: str = Field(..., description="Training precision. (fp32, fp16, bf16, etc.) {string}")
    """Name of the precision to use for training."""
    
    validate_every_n_epochs: float = Field(..., gt=0, description="Validation frequency measured in epochs. Accepts floating point values. {float}")
    """Validation frequency in epochs."""
    
    save_student_every_n_epochs: float = Field(..., gt=0, description="Frequency of saving student model in epochs. {float}")
    """Frequency to save the student model in epochs."""
    
    num_gpu0_layers: int = Field(..., ge=0, description="Number of layers for GPU 0. Used only with device_map = \"custom\". {int}")
    """Number of layers on GPU 0."""
    
    device_map: str = Field(..., description="Device mapping strategy. {string}")
    """Name of the device mapping strategy to use."""
    
    max_memory: Dict[int, str] = Field(..., description="Maximum memory allocation for each device. {dict[str, str]}")
    """Maximum memory allocation for each device."""
    
    multi_gpu: bool = Field(..., description="Whether to do multi-GPU training. {bool}")
    """Flag to enable multi-GPU training."""
    
    save_final_state: bool = Field(..., description="Save the final model state after training. {bool}")
    """Flag to save the final model state."""
    
    wandb_comment: str = Field(..., description="A comment for Weights and Biases logging. {string}")
    """Comment for Weights and Biases logging."""
    
    wandb_project: str = Field(..., description="Weights and Biases project name. {string}")
    """Weights and Biases project name."""
    
    use_flash_attn_2: bool = Field(..., description="Whether to use Flash Attention 2. {bool}")
    """Flag to use Flash Attention 2."""


    # Student settings
    freeze_layers: List[str] = Field(..., description="Layers to freeze during training. {string list}")
    """List of layers to freeze during training."""
    
    add_bos: bool = Field(..., description="Add a beginning-of-sequence token to every sample. {bool}")
    """Flag to add a BOS token to every sample."""
    
    prompt_format: Dict = Field(..., description="Prompt format to use for instruct samples. {JSON}")
    """Prompt format for instruct samples."""


    @field_validator('adam_betas', 'max_memory', 'prompt_format', mode='before')
    def convert_fields(cls, v, info):
        if info.field_name == 'adam_betas' and isinstance(v, list):
            return tuple(v)
        if info.field_name == 'max_memory' and isinstance(v, dict):
            return {int(k): val for k, val in v.items() if k.lower() != 'cpu'}
        if info.field_name == 'prompt_format' and isinstance(v, str):
            return json.loads(v)
        return v

    @model_validator(mode='before')
    def enforce_topk_flag(cls, values):
        if values.get('save_topK') not in (None, 0):
            values['enable_topK'] = True
        return values


def load_json_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}!\nPlease ensure the file exists in the specified location."
        )
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_cli_arguments(parser: argparse.ArgumentParser, model: BaseModel):
    """
    Auto-generate CLI arguments from the Pydantic model's fields.
    """
    for field_name, model_field in model.model_fields.items():
        arg_name = f"--{field_name}"
        field_type = model_field.annotation
        help_text = model_field.description
        kwargs = {"help": help_text, "dest": field_name, "required": False}
        origin = get_origin(field_type)

        # Handle list or tuple types
        if origin in (list, tuple):
            kwargs["nargs"] = "+"
            # Infer element type if available
            args_type = get_args(field_type)
            kwargs["type"] = args_type[0] if args_type else str
        elif field_type == bool:
            # For booleans, use custom conversion function
            kwargs["type"] = str2bool
        else:
            kwargs["type"] = field_type

        parser.add_argument(arg_name, **kwargs)


def merge_config(cli_args: dict, json_config: dict) -> dict:
    """
    Merge CLI arguments with JSON config values. CLI args override JSON config.
    """
    merged = json_config.copy()
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged


def get_config(config_path=None) -> PipelineConfig:
    """
    Loads JSON configuration and merges it with CLI arguments,
    then returns a validated PipelineConfig instance.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.json') if config_path is None else config_path
    json_config = load_json_config(config_path)

    # Check if config.json has all the parameters of the PipelineConfig
    missing_keys = [field for field in PipelineConfig.model_fields if field not in json_config]
    if missing_keys:
        print(f"Error: config.json is missing the following parameters: {', '.join(missing_keys)}")
        print("\nPlease add them to the config file!")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Set parameters for the script.")
    generate_cli_arguments(parser, PipelineConfig)
    args = parser.parse_args()
    cli_args = vars(args)

    merged_params = merge_config(cli_args, json_config)
    return PipelineConfig(**merged_params)


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    config = get_config(config_path=config_path)
    print(config.model_dump_json(indent=4))
