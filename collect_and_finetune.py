from utils.dataset_utils import tokenize_dataset
from classes.teacher.model import TeacherModel
from classes.student.model import StudentModel
from classes.data_manager import H5DataManager
from joblib import Parallel, delayed
from classes.paths import Paths
from tqdm import tqdm
import multiprocessing
import nvidia_smi
import argparse
import signal
import math
import time
import json
import os

nvidia_smi.nvmlInit()
    
def calculate_loop_ids(student: StudentModel, max_cache_size_gb, enable_topK, save_topK):
    """
    Computes loop-based distribution IDs for later usage.
    """
    if enable_topK:
        kb_per_distr_val = (0.001953125 * 3) / 1.7 # storage of one FP16 value * 3 (for int32 index) / 1.7 (avg compression ratio)
        distr_size_kb = kb_per_distr_val * save_topK
    else:
        kb_per_distr_val = 0.001953125 / 1.55 # storage of one FP16 value / 1.55 (avg compression ratio)
        distr_size_kb = kb_per_distr_val * student.crop_to_size
    max_cache_size_kb = max_cache_size_gb * 1e6
    cache_size_kb = 0
    full_collect = False
    loop_ids = []
    chunk_ids = []

    for convo in student.dataset:
        convo_kb = distr_size_kb * convo.len_content
        cache_size_kb += convo_kb
        
        if cache_size_kb >= max_cache_size_kb:
            loop_ids.append(chunk_ids)
            chunk_ids = []
            cache_size_kb = 0
        
        chunk_ids.append(convo.origin_convo_id)
    
    if not loop_ids:
        full_collect = True

    if chunk_ids:
        loop_ids.append(chunk_ids)

    return loop_ids, full_collect


def get_teachers(models_folder, use_teachers: bool) -> list[TeacherModel]:
    """
    Loads teacher models if use_teachers is True, else returns an empty list.
    """
    teachers = []

    if not use_teachers:
        return teachers

    # check if the path is to one model folder or a folder with multiple models
    for file in os.listdir(models_folder):
        if file.endswith(".bin") or file.endswith(".json") or file.endswith(".safetensors"):

            teachers.append(TeacherModel(models_folder))
            return teachers

    model_paths = [os.path.join(models_folder, model_name) for model_name in os.listdir(models_folder)]

    if not model_paths:
        print("No models found in the teacher models folder!")
        exit(0)
    
    for model_path in model_paths:
        teachers.append(TeacherModel(model_path))
    
    return teachers
            

def ensure_compatibility(teachers: list[TeacherModel], student: StudentModel, use_teachers: bool):
    """
    Ensures that teachers and student share tokenization settings before finetuning.
    """
    if not use_teachers:
        return
    
    checklist = {
        'vocab_family': teachers[0].vocab_family,
    }

    for teacher in teachers:
        for key, value in checklist.items():
            if value != getattr(teacher, key):
                raise ValueError(f"Teacher {teacher.model_name} has {key}={getattr(teacher, key)} while the first teacher has {key}={value}")
    
    for key, value in checklist.items():
        if value != getattr(student, key):
            raise ValueError(f"Student has {key} = {getattr(student, key)} while the teachers have {key} = {value}")
        

def prepare_datasets(dataset_path, data_manager: H5DataManager, validation_dataset_path, validation_data_manager: H5DataManager, teachers: list[TeacherModel], student: StudentModel, context_len, save_sys_range, save_user_range, save_assistant_range, ignore_model_type, use_teachers):
    """
    Prepares and tokenizes the training/validation datasets, optionally ignoring certain model types.
    """
    print("Preparing datasets for the student...")

    all_not_add_bos = False
    if not student.add_bos and all(not teacher.add_bos for teacher in teachers):
        all_not_add_bos = True
    
    student.dataset, relevant_ids = tokenize_dataset(
        dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, student, ignore_model_type, all_not_add_bos)

    student.validation_dataset, relevant_ids_val = tokenize_dataset(
        validation_dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, student, ignore_model_type, all_not_add_bos)

    teachers_ids = set()
    teachers_ids_val = set()

    for teacher in teachers:
        print(f"Preparing datasets for {teacher.model_name}...")
        teacher.dataset, teacher_ids = tokenize_dataset(
            dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, teacher, ignore_model_type, all_not_add_bos)

        teacher.validation_dataset, teacher_ids_val = tokenize_dataset(
            validation_dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, teacher, ignore_model_type, all_not_add_bos)
        
        teachers_ids.update(teacher_ids)
        teachers_ids_val.update(teacher_ids_val)

    if use_teachers:
        common_ids = relevant_ids.intersection(teachers_ids)
        common_ids_val = relevant_ids_val.intersection(teachers_ids_val)
    else:
        common_ids = data_manager.get_dataset_ids()
        common_ids_val = validation_data_manager.get_dataset_ids()

    student.dataset = [convo for convo in student.dataset if convo.origin_convo_id in common_ids]
    total_tokens = sum([convo.length for convo in student.dataset])
    total_content_tokens = sum([convo.len_content for convo in student.dataset])
    student.dataset_len = len(student.dataset)
    print(f"Total tokens in dataset: {total_tokens}; Content tokens: {total_content_tokens}")

    student.validation_dataset = [convo for convo in student.validation_dataset if convo.origin_convo_id in common_ids_val]
    total_tokens_val = sum([convo.length for convo in student.validation_dataset])
    total_content_tokens_val = sum([convo.len_content for convo in student.validation_dataset])
    student.validation_dataset_len = len(student.validation_dataset)
    print(f"Total tokens in validation dataset: {total_tokens_val}; Content tokens: {total_content_tokens_val}")

    for teacher in teachers:
        teacher.dataset = [convo for convo in teacher.dataset if convo.origin_convo_id in common_ids]
        teacher.dataset_len = len(teacher.dataset)
        teacher.validation_dataset = [convo for convo in teacher.validation_dataset if convo.origin_convo_id in common_ids_val]
        teacher.validation_dataset_len = len(teacher.validation_dataset)
    

def set_params(teachers: list[TeacherModel], student: StudentModel, crop_to_size: int, context_len: int, temperature: float, device: str, save_topK: int, enable_topK: bool):
    """
    Configures model parameters (e.g., temperature, device).
    """
    for teacher in teachers:
        teacher.crop_to_size = crop_to_size
        teacher.context_len = context_len
        teacher.temperature = temperature
        teacher.device = device
        teacher.topK = save_topK
        teacher.enable_topK = enable_topK

    student.crop_to_size = crop_to_size
    student.context_len = context_len
    student.temperature = temperature
    student.device = device


def set_training_params(student: StudentModel, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_batches, training_precision, decay_start, multi_gpu, data_order, validate_every_n_epochs, 
                        save_student_every_n_epochs, save_final_state, grad_checkpointing, freeze_layers, wandb_comment, alpha, device_map, max_memory, num_gpu0_layers, use_flash_attn_2):
    """
    Configures and schedules all essential training hyperparameters.
    """
    student.num_epochs = num_epochs
    student.eff_batch_size = grad_accum_batches * student.batch_size
    student.num_warmup_steps = math.ceil(num_warmup_steps / student.eff_batch_size)
    student.total_training_steps = num_epochs * student.dataset_len
    student.lr = lr
    student.lr_scheduler_name = lr_scheduler.lower()
    student.optimizer_name = optimizer.lower()
    student.grad_accum = grad_accum_batches
    student.num_grad_accum_batches = math.ceil(student.total_training_steps / student.eff_batch_size)
    student.training_precision_name = training_precision.lower()
    student.decay_start = decay_start
    student.multi_gpu = multi_gpu
    student.data_order = data_order.lower()
    student.validation_every_steps = validate_every_n_epochs * student.dataset_len
    student.next_accum_step = student.eff_batch_size
    student.save_every_steps = save_student_every_n_epochs * student.dataset_len
    student.next_save_step = save_student_every_n_epochs * student.dataset_len
    student.save_final_state = save_final_state
    student.grad_checkpointing = grad_checkpointing
    student.freeze_layers = freeze_layers
    student.use_fa2 = use_flash_attn_2
    student.wandb_comment = wandb_comment
    student.alpha = alpha
    student.device_map_name = device_map
    student.max_memory = max_memory
    student.num_gpu0_layers = num_gpu0_layers


def calculate_sync(student_dataset, data_manager: H5DataManager, student_vocab_family: str):
    """
    Keeps the datasets in sync by removing or renaming mismatched IDs, and telling which IDs to collect.
    """
    def check_shas(disk_sha: dict[str, str], current_sha: dict[str, str]):
        # Create reverse mappings from SHA to set of IDs
        sha_to_ids1 = {}
        sha_to_ids2 = {}

        for id1, sha1 in disk_sha.items():
            sha_to_ids1.setdefault(sha1, set()).add(id1)

        for id2, sha2 in current_sha.items():
            sha_to_ids2.setdefault(sha2, set()).add(id2)

        ids_to_remove = []
        ids_to_collect = []
        ids_to_rename = {}
        
        # Process SHAs to align both datasets
        pbar = tqdm(total=len(disk_sha) + len(current_sha), desc="Checking SHAs", leave=False, postfix="Calculating sync...")
        all_shas = set(sha_to_ids1.keys()).union(set(sha_to_ids2.keys()))
        
        def process_sha(sha):
            local_ids_to_remove = []
            local_ids_to_collect = []
            local_ids_to_rename = {}
            ids1 = sha_to_ids1.get(sha, set())
            ids2 = sha_to_ids2.get(sha, set())

            if len(ids1) > len(ids2):
                surplus = len(ids1) - len(ids2)
                local_ids_to_remove.extend(list(ids1)[:surplus])
                ids1 = ids1 - set(local_ids_to_remove)

            if len(ids2) > len(ids1):
                missing = len(ids2) - len(ids1)
                local_ids_to_collect.extend(list(ids2)[:missing])
                ids2 = ids2 - set(local_ids_to_collect)

            ids1 = list(ids1)
            ids2 = list(ids2)
            for id1, id2 in zip(ids1, ids2):
                if id1 != int(id2):
                    local_ids_to_rename[id1] = id2

            return local_ids_to_collect, local_ids_to_rename, local_ids_to_remove

        results = Parallel(n_jobs=-1, prefer='threads')(delayed(process_sha)(sha) for sha in all_shas)
        
        for collect, rename, remove in results:
            ids_to_collect.extend(collect)
            ids_to_rename.update(rename)
            ids_to_remove.extend(remove)
            pbar.update()
        
        pbar.close()

        return ids_to_collect, ids_to_rename, ids_to_remove

    def get_collection_info(current_shas: dict[str: str], data_manager: H5DataManager, student_vocab_family: str):
        disk_shas = data_manager.get_available_shas()
        data_manager_vocab_family = data_manager.get_vocab_family()

        if data_manager_vocab_family is None:
            return list(current_shas.keys()), {}, list(disk_shas.keys())

        if not data_manager_vocab_family == student_vocab_family:
            print("Vocab family mismatch between the current models and the dataset on disk!\nAll data will be recollected using the new vocab family.")
            return list(current_shas.keys()), {}, list(disk_shas.keys())

        return check_shas(disk_shas, current_shas)

    current_shas:dict[str, str] = {f"{convo.origin_convo_id}": convo.content_sha for convo in student_dataset}

    ids_collect, ids_rename, ids_remove = get_collection_info(current_shas, data_manager, student_vocab_family)

    ids_collect = [int(id) for id in ids_collect]
    ids_remove = [int(id) for id in ids_remove]

    return ids_collect, ids_rename, ids_remove


def sync_datasets(validation_data_manager: H5DataManager, data_manager: H5DataManager, rebase, student: StudentModel, use_teachers: bool):
    if rebase:
        current_shas = {f"{convo.origin_convo_id}": convo.content_sha for convo in student.dataset}
        current_shas_val = {f"{convo.origin_convo_id}": convo.content_sha for convo in student.validation_dataset}

        data_manager.update_shas(current_shas)
        validation_data_manager.update_shas(current_shas_val)

        data_manager.set_vocab_family(student.vocab_family)
        validation_data_manager.set_vocab_family(student.vocab_family)
        return [], []
    
    if not use_teachers:
        return [], []
    
    ids_collect, ids_rename, ids_remove = calculate_sync(student.dataset, data_manager, student.vocab_family)
    ids_collect_val, ids_rename_val, ids_remove_val = calculate_sync(student.validation_dataset, validation_data_manager, student.vocab_family)

    data_manager.sync(ids_remove, ids_rename)
    validation_data_manager.sync(ids_remove_val, ids_rename_val)

    return ids_collect, ids_collect_val


def check_topk(data_manager: H5DataManager, enable_topk, save_topk: int, validation=False):
    if not enable_topk:
        return save_topk
    
    dataset_topk = data_manager.get_topk()

    if dataset_topk is None:
        data_manager.set_topk(save_topk)
        return save_topk
    
    text_insert = "validation" if validation else "main"

    if dataset_topk != save_topk:
        print(f"\nTopK mismatch between the {text_insert} h5 dataset, and your config!\n{text_insert.upper()} dataset has TopK={dataset_topk}, while you set TopK={save_topk}.")
        response = input("Do you wish to continue with the dataset's TopK?\nIf not, the program will exit to allow you to change the config. (y/n): ")

        if response.lower() not in ["y", "ye", "yes", "1", "true", "t"]:
            handle_termination(None, None)
        
        return dataset_topk

    return save_topk
        

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}!\nPlease make sure the file exists, and has all the necessary parameters.")
    with open(config_path, 'r') as f:
        return json.load(f)



def get_params():
    parser = argparse.ArgumentParser(description="Set parameters for the script.", formatter_class=argparse.RawTextHelpFormatter)
    
    # Groups
    path_group = parser.add_argument_group('Path args')
    cache_group = parser.add_argument_group('Cache args')
    pipeline_group = parser.add_argument_group('Pipeline args')
    general_model_group = parser.add_argument_group('General model args')
    collection_group = parser.add_argument_group('Data collection args')
    training_group = parser.add_argument_group('Training args')
    student_group = parser.add_argument_group('Student args')

    # Paths
    path_group.add_argument('--cache_folder', '-c', type=str, help='Directory for cache storage.\nIdeally should be an empty folder. {string}')
    path_group.add_argument('--dataset_path', '-d', type=str, help='Path to the training dataset. {string}')
    path_group.add_argument('--validation_dataset_path', '-vd', type=str, help='Path to the validation dataset. {string}')
    path_group.add_argument('--teacher_models_folder', '-tm', type=str, help='Directory containing teacher models,\nor a path to just one teacher directly. {string}')
    path_group.add_argument('--student_path', '-s', type=str, help='Path to the student model. {string}')

    # Cache settings
    cache_group.add_argument('--max_cache_size_gb', '-maxgb', type=float, help='Maximum cache size in GB.\nUsed to keep the main h5 dataset under this limit,\nand use chunked collection+training when the calculated size of the collected h5 dataset is over this limit.\nOnly tracks the main h5 dataset\'s size, any misc. files/states are not counted. {float}')

    # Pipeline settings
    pipeline_group.add_argument('--ignore_model_type', type=bool, help='If True, will let completion teachers collect instruct data,\nand instruct teachers completion data.\nUse at your own discretion. {bool}')
    pipeline_group.add_argument('--rebase_dataset', type=bool, help='Rebase the dataset without safety checks.\nOverwrites all metadata in the h5 dataset,\nso be very careful to use the exact same text dataset as the one used for its collection.\nIntended to update the dataset to a newer version of the pipeline after breaking changes. {bool}')
    pipeline_group.add_argument('--use_teachers', type=bool, help='Whether to use teachers for distillation.\nUseful when you downloaded the dataset from the internet and want to distill using it,\nbut don\'t have the teachers it was collected with on disk. {bool}')
    
    # General model settings
    general_model_group.add_argument('--context_len', '-ctx', type=int, help='Context length to collect and train on. {int}')
    general_model_group.add_argument('--save_sys_range', type=bool, help='Boolean flag to save specific token ranges within conversations.\nUsed only with instruct data to collect and train only on the content of the conversation,\navoiding the prompt formatting tokens. {bool}')
    general_model_group.add_argument('--save_user_range', type=bool, help='Boolean flag to save specific token ranges within conversations.\nUsed only with instruct data to collect and train only on the content of the conversation,\navoiding the prompt formatting tokens. {bool}')
    general_model_group.add_argument('--save_assistant_range', type=bool, help='Boolean flag to save specific token ranges within conversations.\nUsed only with instruct data to collect and train only on the content of the conversation,\navoiding the prompt formatting tokens. {bool}')
    general_model_group.add_argument('--crop_distr_to_size', type=int, help='Crop distribution size for token filtering.\nA rudimentary way to avoid training on tokens that aren\'t in the student\'s vocab.\nApplied as: distributions[:, :crop_distr_to_size],\nwhere the first dimension is for tokens, and the second for logits.\nMust be set to the base-model\'s vocabulary size. {int}')
    general_model_group.add_argument('--enable_topK', type=bool, help='Enable top-K sampling for collecting and training. {bool}')
    general_model_group.add_argument('--save_topK', '-topk', type=int, help='Configure top-K sampling for collecting and training.\nSlashes storage requirements enormously,\nwithout it, 1000 2048-token long samples take up ~100gb of storage at vocabulary size of 32000,\nbut with top-K at 200, this goes down to only ~1.4gb. {int}')
    general_model_group.add_argument('--device', type=str, help='Main device for any single-device tensor operations (e.g., cuda:0). {string}')

    # Collection settings
    collection_group.add_argument('--num_inference_workers', '-niw', type=int, help='Number of inference workers for parallel collection.\nUse if you have enough VRAM.\nNote: it can\'t handle more than 3 inference workers due to multiprocessing issues,\nelse it just hangs indefinitely. {int}')
    collection_group.add_argument('--reserve_vram', type=list, nargs='+', help='Amount of VRAM to reserve per GPU during collection,\nwill try to keep that much memory in GB free for each GPU.\nExample: [4, 0.5] for 4GB reserved on the first GPU, and 0.5GB on the second. {float list}')

    # Training settings
    training_group.add_argument('--num_epochs', '-ne', type=int, help='Number of training epochs. {int}')
    training_group.add_argument('--num_warmup_steps', '-nws', type=int, help='Number of warmup steps for learning rate. {int}')
    training_group.add_argument('--batch_size', '-bs', type=int, help='Training batch size. {int}')
    training_group.add_argument('--grad_accum_batches', '-g', type=int, help='Number of gradient accumulations done before calling optimizer.step(). {int}')
    training_group.add_argument('--grad_checkpointing', type=bool, help='Enable gradient checkpointing for memory savings,\nbut slower training. {bool}')
    training_group.add_argument('--temperature', '-t', type=float, help='Temperature for distillation. {float}')
    training_group.add_argument('--lr', '-lr', type=float, help='Learning rate. {float}')
    training_group.add_argument('--decay_start', type=float, help='Start decaying learning rate to 0 at this percentage of total training steps,\nonly used by the wsd learning rate scheduler. {float}')
    training_group.add_argument('--alpha', type=float, help='Weighting factor for weighted losses.\nCurrently used as: weights = ((kl_div_per_token / kl_div_per_token.max()) + 1).pow(alpha). {float}')
    training_group.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler name.\nOptions include a custom implementation of Warmup Stable Decay learning rate scheduler from MiniCPM,\nand any other learning rate scheduler from Transformer\'s get_scheduler() function (e.g., cosine, linear, constant). {string}')
    training_group.add_argument('--optimizer', type=str, help='Optimizer name.\nCurrently available options include: adam, adamw, adamw8bit, adamw32bit,\npaged_adamw, paged_adamw8bit, paged_adamw32bit, sgd, rmsprop, rmsprop8bit, rmsprop32bit, adagrad. {string}')
    training_group.add_argument('--data_order', type=str, help='Order of samples during training.\nOptions are: shuffle - randomizes order of samples every epoch,\nsorted - sort the data by length of samples,\nnative - same order of samples as in the text dataset. {string}')
    training_group.add_argument('--training_precision', type=str, help='Training precision.\nOptions include: fp32, fp16, bf16 (also supports bnb\'s 4bit and 8bit,\nbut they are very buggy at the moment, use at your own discretion). {string}')
    training_group.add_argument('--validate_every_n_epochs', type=float, help='Validation frequency measured in epochs to scale with your dataset.\nAccepts floating point numbers like 0.1,\nthis will do 10 validation steps every epoch, etc. {float}')
    training_group.add_argument('--save_student_every_n_epochs', type=float, help='Frequency of saving student model in epochs.\nSame as above. {float}')
    training_group.add_argument('--num_gpu0_layers', type=int, help='Number of layers for GPU 0.\nUsed only with device_map = "custom". {int}')
    training_group.add_argument('--device_map', type=str, help='Device mapping strategy.\nCurrently supports any device map provided by HF accelerate: balanced, balanced_low_0,\nand custom for our custom splitting strategy,\nalso allowing you to set how many layers you want on the first GPU with num_gpu0_layers. {string}')
    training_group.add_argument('--max_memory', type=dict, help='Maximum memory allocation for each device.\nMust be formatted as a Dict, an example is provided within the config.json in the repo.\nEnsure it is edited according to the number of GPUs you have. {dict[str, str]}')
    training_group.add_argument('--multi_gpu', type=bool, help='Whether to do multi-GPU training. {bool}')
    training_group.add_argument('--save_final_state', type=bool, help='Save the final model state after training.\nCan consume enormous amounts of storage and RAM. {bool}')
    training_group.add_argument('--wandb_comment', '-wdb', type=str, help='A comment for Weights and Biases logging.\nExample: {comment} ModelName lr(lr) (Date/Time). {string}')
    training_group.add_argument('--use_flash_attn_2', '-fa2', type=bool, help='Whether to use Flash Attention 2,\nor default to sdpa. {bool}')

    # Student settings
    student_group.add_argument('--freeze_layers', '-fl', type=str, nargs='+', help='Layers to freeze during training.\nUses a list of strings for layer names to freeze.\nExample: [".block_sparse_moe.gate", ...] {string list}')
    student_group.add_argument('--add_bos', type=bool, help='Add beginning-of-sequence token to every sample. {bool}')
    student_group.add_argument('--prompt_format', type=json.loads, help='Prompt format to use for instruct samples. {JSON}')


    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = load_config(config_path)

    params = {key: getattr(args, key) if getattr(args, key) is not None else config[key] for key in config.keys()}

    if getattr(args, "save_topK") is not None:
        params["enable_topK"] = True

    params['max_memory'] = {int(key): value for key, value in params['max_memory'].items() if key.lower() != 'cpu'}

    return params


def handle_termination(signum, frame):
    print("\nTerminating the script...")
    if data_manager is not None:
        data_manager.close()

    if validation_data_manager is not None:
        validation_data_manager.close()

    if teachers is not None:
        for teacher in teachers:
            teacher.close()

    if student is not None:
        student.close()

    os._exit(0)


def update_teachers_param(teachers, param, value):
    for teacher in teachers:
        setattr(teacher, param, value)


def main():
    params = get_params()

    # Paths
    cache_folder = params['cache_folder']
    dataset_path = params['dataset_path']
    validation_dataset_path = params['validation_dataset_path']
    teacher_models_folder = params['teacher_models_folder']
    student_path = params['student_path']

    # Cache settings
    max_cache_size_gb = params['max_cache_size_gb']

    # Pipeline settings
    ignore_model_type = params['ignore_model_type'] # If True, will collect all data from teachers regardless if both, conversation and teacher are matching in being completion/instruct
    rebase_dataset = params['rebase_dataset'] # Only use if you know what you are doing. Will skip safety checks to directly use data on disk, also updating metadata used for future safety checks
    use_teachers = params['use_teachers'] # If True, will use the teacher models to collect data for the student if its needed, and will run safety checks for them

    # General model settings
    context_len = params['context_len']
    save_sys_range = params['save_sys_range']
    save_user_range = params['save_user_range']
    save_assistant_range = params['save_assistant_range']
    crop_distr_to_size = params['crop_distr_to_size'] # A very rudimentary way to avoid training on tokens that aren't in the student's vocab. Its applied as: distributions[:, :crop_distr_to_size] 
    enable_topK = params['enable_topK']
    save_topK = params['save_topK']
    device = params['device']

    # Collection settings
    num_inference_workers = params['num_inference_workers']
    reserve_vram = params['reserve_vram']

    # Training settings
    num_epochs = params['num_epochs']
    num_warmup_steps = params['num_warmup_steps']
    batch_size = params['batch_size']
    grad_accum_batches = params['grad_accum_batches']
    grad_checkpointing = params['grad_checkpointing']
    temperature = params['temperature']
    lr = params['lr']
    decay_start = params['decay_start'] # Start decay at this % of total steps trained
    alpha = params['alpha'] # weighted losses
    lr_scheduler = params['lr_scheduler'] # "wsd", "cosine", "linear", "constant"
    optimizer = params['optimizer'] # "adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit", "sgd", "rmsprop", "rmsprop8bit", "rmsprop32bit", "adagrad"
    data_order = params['data_order'] # "shuffle", "native", "sorted"
    training_precision = params['training_precision'] # "fp32", "fp16", "bf16", "4bit", "8bit"
    validate_every_n_epochs = params['validate_every_n_epochs']
    save_student_every_n_epochs = params['save_student_every_n_epochs']
    num_gpu0_layers = params['num_gpu0_layers'] # Used only if `device_map` is set to "custom"
    device_map = params['device_map'] # "balanced", "balanced_low_0", "custom"
    max_memory = params['max_memory'] # {0: '20GB', 1: '60GB', 2: '60GB', 3: '60GB', 'cpu': '200GB'}
    multi_gpu = params['multi_gpu']
    save_final_state = params['save_final_state'] # Save the final state of the model after training
    use_flash_attn_2 = params['use_flash_attn_2']
    wandb_comment = params['wandb_comment'] # Comment for wandb: {comment} ModelName lr(lr) (Date/Time)

    # Student settings
    freeze_layers = params['freeze_layers']
    add_bos = params['add_bos']
    prompt_format = params['prompt_format']


    # Initialization
    global data_manager, validation_data_manager, teachers, student

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)

    multiprocessing.set_start_method('spawn', force=True)
    paths = Paths(cache_folder)
    teachers = get_teachers(teacher_models_folder, use_teachers)
    student = StudentModel(student_path, paths, add_bos, prompt_format, batch_size)

    print("Launching data managers...")
    data_manager = H5DataManager(paths.dataset, device, manager_name="main")
    validation_data_manager = H5DataManager(paths.dataset_validation, device, manager_name="validation")
    
    ensure_compatibility(teachers, student, use_teachers)
    prepare_datasets(dataset_path, data_manager, validation_dataset_path, validation_data_manager, teachers, student, context_len, save_sys_range, save_user_range, save_assistant_range,
                     ignore_model_type, use_teachers)

    set_params(teachers, student, crop_distr_to_size, context_len, temperature, device, save_topK, enable_topK)
    set_training_params(student, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_batches, training_precision, decay_start, multi_gpu, data_order, validate_every_n_epochs, 
                        save_student_every_n_epochs, save_final_state, grad_checkpointing, freeze_layers, wandb_comment, alpha, device_map, max_memory, num_gpu0_layers, use_flash_attn_2)

    student.reorder_dataset()
    
    loop_ids, full_collect = calculate_loop_ids(student, max_cache_size_gb, enable_topK, save_topK)
    
    ids_collect, ids_collect_val = sync_datasets(validation_data_manager, data_manager, rebase_dataset, student, use_teachers)

    # Main logic

    ## Validation collection
    if ids_collect_val and not rebase_dataset:
        print(f"Collecting validation data for {len(ids_collect_val)} samples...")

        topk_to_use = check_topk(validation_data_manager, enable_topK, save_topK, validation=True)
        update_teachers_param(teachers, "topK", topk_to_use)

        validation_data_manager.set_vocab_family(student.vocab_family)

        for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False, disable=len(teachers) == 1):
            teacher.process_chunk(reserve_vram, num_inference_workers, data_manager=validation_data_manager, ids_to_collect=ids_collect_val, validation=True)
    
    else:
        print("Using data on disk for validation...")
    
    
    ## Training collection and finetuning
    if ids_collect and not rebase_dataset:

        data_manager.set_vocab_family(student.vocab_family)

        if full_collect:
            print(f"Collecting data for {len(ids_collect)} samples...")

            topk_to_use = check_topk(data_manager, enable_topK, save_topK)
            update_teachers_param(teachers, "topK", topk_to_use)

            for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False, disable=len(teachers) == 1):
                teacher.process_chunk(reserve_vram, num_inference_workers, ids_to_collect=ids_collect, data_manager=data_manager)

            student.train_chunk(data_manager, validation_data_manager, full_collect)

        else:
            print(f"WARNING: The calculated size of the h5 dataset is too large for the specified cache size ({max_cache_size_gb}GB).")
            print("The data will be collected and trained on in chunks.")

            if data_manager.has_data():
                print("This will erase all previous data in the main h5 dataset!")
                response = input("Do you wish to continue? (y/n): ")

                if response.lower() not in ["y", "ye", "yes", "1", "true", "t"]:
                    handle_termination(None, None)

            data_manager.set_topk(save_topK)

            for epoch in tqdm(range(num_epochs), desc="Epochs", smoothing=0.06, position=0):

                for chunk_ids in tqdm(loop_ids, desc="Chunks", smoothing=0.06, position=1, leave=False):
                    data_manager.purge_dataset(ask_confirmation=False)

                    for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=2, leave=False):
                        teacher.process_chunk(reserve_vram, num_inference_workers, ids_to_collect=chunk_ids, data_manager=data_manager)

                    student.train_chunk(data_manager, validation_data_manager, full_collect)

    else:
        print("Using data on disk for training...")

        student.train_chunk(data_manager, validation_data_manager, True)

    print("Done!")

    handle_termination(None, None)

if __name__ == "__main__":
    main()
