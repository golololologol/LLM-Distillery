import argparse
import json
import os

class PathArgs:
    def __init__(self, cache_folder: str, dataset_path: str, validation_dataset_path: str, teacher_models_folder: str, student_path: str):
        self.cache_folder = cache_folder
        self.dataset_path = dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.teacher_models_folder = teacher_models_folder
        self.student_path = student_path

class PipelineArgs:
    def __init__(self, max_cache_size_gb: float, ignore_model_type: bool, rebase_dataset: bool, use_teachers: bool):
        self.max_cache_size_gb = max_cache_size_gb
        self.ignore_model_type = ignore_model_type
        self.rebase_dataset = rebase_dataset
        self.use_teachers = use_teachers

class ModelArgs:
    def __init__(self, context_len: int, save_sys_range: bool, save_user_range: bool, save_assistant_range: bool,
                 crop_distr_to_size: int, enable_topK: bool, save_topK: int, device: str):
        self.context_len = context_len
        self.save_sys_range = save_sys_range
        self.save_user_range = save_user_range
        self.save_assistant_range = save_assistant_range
        self.crop_distr_to_size = crop_distr_to_size
        self.enable_topK = enable_topK
        self.save_topK = save_topK
        self.device = device

class CollectionArgs:
    def __init__(self, num_inference_workers: int, reserve_vram: list):
        self.num_inference_workers = num_inference_workers
        self.reserve_vram = reserve_vram

class TrainingArgs:
    def __init__(self, num_epochs: int, num_warmup_steps: int, batch_size: int, grad_accum_batches: int, grad_checkpointing: bool,
                 temperature: float, lr: float, adam_betas: tuple, adam_decay: float, lr_decay_start: float, alpha: float,
                 lr_scheduler: str, optimizer: str, data_order: str, training_precision: str, validate_every_n_epochs: float,
                 save_student_every_n_epochs: float, num_gpu0_layers: int, device_map: str, max_memory: dict, multi_gpu: bool,
                 save_final_state: bool, wandb_comment: str, wandb_project: str, use_flash_attn_2: bool):
        self.num_epochs = num_epochs
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.grad_accum_batches = grad_accum_batches
        self.grad_checkpointing = grad_checkpointing
        self.temperature = temperature
        self.lr = lr
        self.adam_betas = adam_betas
        self.adam_decay = adam_decay
        self.lr_decay_start = lr_decay_start
        self.alpha = alpha
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.data_order = data_order
        self.training_precision = training_precision
        self.validate_every_n_epochs = validate_every_n_epochs
        self.save_student_every_n_epochs = save_student_every_n_epochs
        self.num_gpu0_layers = num_gpu0_layers
        self.device_map = device_map
        self.max_memory = max_memory
        self.multi_gpu = multi_gpu
        self.save_final_state = save_final_state
        self.wandb_comment = wandb_comment
        self.wandb_project = wandb_project
        self.use_flash_attn_2 = use_flash_attn_2

class StudentArgs:
    def __init__(self, freeze_layers: list, add_bos: bool, prompt_format: dict):
        self.freeze_layers = freeze_layers
        self.add_bos = add_bos
        self.prompt_format = prompt_format

class Config:
    def __init__(self, path_args: PathArgs, pipeline_args: PipelineArgs,
                 model_args: ModelArgs, collection_args: CollectionArgs, training_args: TrainingArgs, student_args: StudentArgs):
        self.paths = path_args
        self.pipeline = pipeline_args
        self.model = model_args
        self.collection = collection_args
        self.training = training_args
        self.student = student_args

def load_config_args():
    parser = argparse.ArgumentParser(
        description="Set parameters for the script.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ----- Path Arguments -----
    path_group = parser.add_argument_group('Path args')
    path_group.add_argument('--cache_folder', '-c', type=str, help='Directory for cache storage.')
    path_group.add_argument('--dataset_path', '-d', type=str, help='Path to the training dataset.')
    path_group.add_argument('--validation_dataset_path', '-vd', type=str, help='Path to the validation dataset.')
    path_group.add_argument('--teacher_models_folder', '-tm', type=str, help='Directory containing teacher models.')
    path_group.add_argument('--student_path', '-s', type=str, help='Path to the student model.')

    # ----- Cache Arguments -----
    cache_group = parser.add_argument_group('Cache args')
    cache_group.add_argument('--max_cache_size_gb', '-maxgb', type=float, help='Maximum cache size in GB.')

    # ----- Pipeline Arguments -----
    pipeline_group = parser.add_argument_group('Pipeline args')
    pipeline_group.add_argument('--ignore_model_type', type=bool, help='If True, ignore model type.')
    pipeline_group.add_argument('--rebase_dataset', type=bool, help='Rebase the dataset without safety checks.')
    pipeline_group.add_argument('--use_teachers', type=bool, help='Whether to use teacher models.')

    # ----- General Model Arguments -----
    model_group = parser.add_argument_group('General model args')
    model_group.add_argument('--context_len', '-ctx', type=int, help='Context length.')
    model_group.add_argument('--save_sys_range', type=bool, help='Save system token range.')
    model_group.add_argument('--save_user_range', type=bool, help='Save user token range.')
    model_group.add_argument('--save_assistant_range', type=bool, help='Save assistant token range.')
    model_group.add_argument('--crop_distr_to_size', type=int, help='Crop distribution size.')
    model_group.add_argument('--enable_topK', type=bool, help='Enable top-K sampling.')
    model_group.add_argument('--save_topK', '-topk', type=int, help='Top-K value.')
    model_group.add_argument('--device', type=str, help='Device to use.')

    # ----- Collection Arguments -----
    collection_group = parser.add_argument_group('Collection args')
    collection_group.add_argument('--num_inference_workers', '-niw', type=int, help='Number of inference workers.')
    collection_group.add_argument('--reserve_vram', type=float, nargs='+', help='Amount of VRAM to reserve per GPU.')

    # ----- Training Arguments -----
    training_group = parser.add_argument_group('Training args')
    training_group.add_argument('--num_epochs', '-ne', type=int, help='Number of training epochs.')
    training_group.add_argument('--num_warmup_steps', '-nws', type=int, help='Number of warmup steps.')
    training_group.add_argument('--batch_size', '-bs', type=int, help='Training batch size.')
    training_group.add_argument('--grad_accum_batches', '-g', type=int, help='Gradient accumulation batches.')
    training_group.add_argument('--grad_checkpointing', type=bool, help='Enable gradient checkpointing.')
    training_group.add_argument('--temperature', '-t', type=float, help='Temperature.')
    training_group.add_argument('--lr', '-lr', type=float, help='Learning rate.')
    training_group.add_argument('--adam_betas', '-ab', type=float, nargs='+', help='Adam betas.')
    training_group.add_argument('--adam_decay', type=float, help='Adam decay.')
    training_group.add_argument('--lr_decay_start', type=float, help='LR decay start percentage.')
    training_group.add_argument('--alpha', type=float, help='Alpha weighting factor.')
    training_group.add_argument('--lr_scheduler', type=str, help='LR scheduler name.')
    training_group.add_argument('--optimizer', type=str, help='Optimizer name.')
    training_group.add_argument('--data_order', type=str, help='Order of data samples.')
    training_group.add_argument('--training_precision', type=str, help='Training precision.')
    training_group.add_argument('--validate_every_n_epochs', type=float, help='Validation frequency in epochs.')
    training_group.add_argument('--save_student_every_n_epochs', type=float, help='Student saving frequency in epochs.')
    training_group.add_argument('--num_gpu0_layers', type=int, help='Number of layers for GPU 0.')
    training_group.add_argument('--device_map', type=str, help='Device mapping strategy.')
    training_group.add_argument('--max_memory', type=json.loads, help='Maximum memory allocation as a JSON dict.')
    training_group.add_argument('--multi_gpu', type=bool, help='Enable multi-GPU training.')
    training_group.add_argument('--save_final_state', type=bool, help='Save final model state.')
    training_group.add_argument('--wandb_comment', '-wdb', type=str, help='Wandb comment.')
    training_group.add_argument('--wandb_project', type=str, help='Wandb project name.')
    training_group.add_argument('--use_flash_attn_2', '-fa2', type=bool, help='Use Flash Attention 2.')

    # ----- Student Arguments -----
    student_group = parser.add_argument_group('Student args')
    student_group.add_argument('--freeze_layers', '-fl', type=str, nargs='+', help='List of layers to freeze.')
    student_group.add_argument('--add_bos', type=bool, help='Add beginning-of-sequence token.')
    student_group.add_argument('--prompt_format', type=json.loads, help='Prompt format as JSON.')

    args = parser.parse_args()

    # Load file-based defaults if available
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    file_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)

    # Utility to return the command-line argument if given, else fallback to the config file default
    def get_arg(key, default=None):
        return getattr(args, key, None) if getattr(args, key, None) is not None else file_config.get(key, default)

    path_args = PathArgs(
        cache_folder=get_arg('cache_folder'),
        dataset_path=get_arg('dataset_path'),
        validation_dataset_path=get_arg('validation_dataset_path'),
        teacher_models_folder=get_arg('teacher_models_folder'),
        student_path=get_arg('student_path')
    )

    pipeline_args = PipelineArgs(
        max_cache_size_gb=get_arg('max_cache_size_gb'),
        ignore_model_type=get_arg('ignore_model_type'),
        rebase_dataset=get_arg('rebase_dataset'),
        use_teachers=get_arg('use_teachers')
    )

    model_args = ModelArgs(
        context_len=get_arg('context_len'),
        save_sys_range=get_arg('save_sys_range'),
        save_user_range=get_arg('save_user_range'),
        save_assistant_range=get_arg('save_assistant_range'),
        crop_distr_to_size=get_arg('crop_distr_to_size'),
        enable_topK=get_arg('enable_topK'),
        save_topK=get_arg('save_topK'),
        device=get_arg('device')
    )

    collection_args = CollectionArgs(
        num_inference_workers=get_arg('num_inference_workers'),
        reserve_vram=get_arg('reserve_vram')
    )

    adam_betas_val = tuple(get_arg('adam_betas')) if get_arg('adam_betas') is not None else None
    training_args = TrainingArgs(
        num_epochs=get_arg('num_epochs'),
        num_warmup_steps=get_arg('num_warmup_steps'),
        batch_size=get_arg('batch_size'),
        grad_accum_batches=get_arg('grad_accum_batches'),
        grad_checkpointing=get_arg('grad_checkpointing'),
        temperature=get_arg('temperature'),
        lr=get_arg('lr'),
        adam_betas=adam_betas_val,
        adam_decay=get_arg('adam_decay'),
        lr_decay_start=get_arg('lr_decay_start'),
        alpha=get_arg('alpha'),
        lr_scheduler=get_arg('lr_scheduler'),
        optimizer=get_arg('optimizer'),
        data_order=get_arg('data_order'),
        training_precision=get_arg('training_precision'),
        validate_every_n_epochs=get_arg('validate_every_n_epochs'),
        save_student_every_n_epochs=get_arg('save_student_every_n_epochs'),
        num_gpu0_layers=get_arg('num_gpu0_layers'),
        device_map=get_arg('device_map'),
        max_memory=get_arg('max_memory'),
        multi_gpu=get_arg('multi_gpu'),
        save_final_state=get_arg('save_final_state'),
        wandb_comment=get_arg('wandb_comment'),
        wandb_project=get_arg('wandb_project'),
        use_flash_attn_2=get_arg('use_flash_attn_2')
    )

    student_args = StudentArgs(
        freeze_layers=get_arg('freeze_layers'),
        add_bos=get_arg('add_bos'),
        prompt_format=get_arg('prompt_format')
    )

    return Config(path_args, pipeline_args, model_args, collection_args, training_args, student_args)
