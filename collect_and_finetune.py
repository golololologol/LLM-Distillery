from utils.dataset_utils import tokenize_dataset
from classes.teacher.model import TeacherModel
from classes.student.model import StudentModel
from classes.data_manager import H5DataManager
from classes.paths import Paths
from tqdm import tqdm
import multiprocessing
import nvidia_smi
import argparse
import math
import time
import json
import os

nvidia_smi.nvmlInit()
    
def calculate_loop_ids(student: StudentModel, max_cache_size_gb, enable_topK, save_topK):
    
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


def get_teachers(models_folder) -> list[TeacherModel]:
    teachers = []

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
            

def ensure_compatibility(teachers: list[TeacherModel], student: StudentModel):
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
        

def prepare_datasets(dataset_path, validation_dataset_path, teachers: list[TeacherModel], student: StudentModel, context_len, save_sys_range, save_user_range, save_assistant_range, ignore_model_type):
    print("Preparing datasets for the student...")
    student.dataset, relevant_ids = tokenize_dataset(
        dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, student, ignore_model_type)

    student.validation_dataset, relevant_ids_val = tokenize_dataset(
        validation_dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, student, ignore_model_type)

    teachers_ids = set()
    teachers_ids_val = set()

    for teacher in teachers:
        print(f"Preparing datasets for {teacher.model_name}...")
        teacher.dataset, teacher_ids = tokenize_dataset(
            dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, teacher, ignore_model_type)

        teacher.validation_dataset, teacher_ids_val = tokenize_dataset(
            validation_dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, teacher, ignore_model_type)
        
        teachers_ids.update(teacher_ids)
        teachers_ids_val.update(teacher_ids_val)

    common_ids = relevant_ids.intersection(teachers_ids)
    common_ids_val = relevant_ids_val.intersection(teachers_ids_val)

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
    

def set_params(teachers: list[TeacherModel], student: StudentModel, crop_to_size: int, context_len: int, temperature: float, device: str, save_topK: int, enable_topK: bool, encourage_eos: bool):
    for teacher in teachers:
        teacher.crop_to_size = crop_to_size
        teacher.context_len = context_len
        teacher.temperature = temperature
        teacher.student_eos_id = student.special_tokens["eos_id"]
        teacher.device = device
        teacher.topK = save_topK
        teacher.enable_topK = enable_topK
        teacher.encourage_eos = encourage_eos

    student.crop_to_size = crop_to_size
    student.context_len = context_len
    student.temperature = temperature
    student.device = device


def set_training_params(student: StudentModel, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_batches, training_precision, decay_start, multi_gpu, data_order,
                        validate_every_n_epochs, save_student_every_n_epochs, save_final_state, grad_checkpointing, freeze_layers, wandb_comment, alpha, device_map, max_memory, num_gpu0_layers):
    
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
    student.wandb_comment = wandb_comment
    student.alpha = alpha
    student.device_map_name = device_map
    student.max_memory = max_memory
    student.num_gpu0_layers = num_gpu0_layers


def calculate_sync(student_dataset, data_manager: H5DataManager, student_vocab_family: str):
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
        all_shas = set(sha_to_ids1.keys()).union(set(sha_to_ids2.keys()))
        for sha in all_shas:
            ids1 = sha_to_ids1.get(sha, set())
            ids2 = sha_to_ids2.get(sha, set())

            if len(ids1) > len(ids2):
                surplus = len(ids1) - len(ids2)
                ids_to_remove.extend(list(ids1)[:surplus])
                ids1 = ids1 - set(ids_to_remove)

            if len(ids2) > len(ids1):
                missing = len(ids2) - len(ids1)
                ids_to_collect.extend(list(ids2)[:missing])
                ids2 = ids2 - set(ids_to_collect)

            ids1 = list(ids1)
            ids2 = list(ids2)
            for id1, id2 in zip(ids1, ids2):
                if id1 != id2:
                    ids_to_rename[id1] = id2

        return ids_to_collect, ids_to_rename, ids_to_remove

    def get_collection_info(current_shas, data_manager: H5DataManager, student_vocab_family: str):
        disk_shas = data_manager.get_available_shas()
        data_manager_vocab_family = data_manager.get_vocab_family()

        if data_manager_vocab_family is None:
            return list(current_shas.keys()), {}, list(disk_shas.keys())

        if not data_manager_vocab_family == student_vocab_family:
            print("Vocab family mismatch between the current models and the dataset on disk!\nAll data will be recollected using the new vocab family.")
            return list(current_shas.keys()), {}, list(disk_shas.keys())

        return check_shas(disk_shas, current_shas)

    current_shas = {f"{convo.origin_convo_id}": convo.content_sha for convo in student_dataset}

    ids_collect, ids_rename, ids_remove = get_collection_info(current_shas, data_manager, student_vocab_family)

    ids_collect = [int(id) for id in ids_collect]
    ids_remove = [int(id) for id in ids_remove]

    return ids_collect, ids_rename, ids_remove


def sync_datasets(validation_data_manager: H5DataManager, data_manager: H5DataManager, rebase, student: StudentModel):
    if rebase:
        current_shas = {f"{convo.origin_convo_id}": convo.content_sha for convo in student.dataset}
        current_shas_val = {f"{convo.origin_convo_id}": convo.content_sha for convo in student.validation_dataset}
        data_manager.update_shas(current_shas)
        validation_data_manager.update_shas(current_shas_val)
        data_manager.set_vocab_family(student.vocab_family)
        validation_data_manager.set_vocab_family(student.vocab_family)
        return [], []
    
    ids_collect, ids_rename, ids_remove = calculate_sync(student.dataset, data_manager, student.vocab_family)
    ids_collect_val, ids_rename_val, ids_remove_val = calculate_sync(student.validation_dataset, validation_data_manager, student.vocab_family)

    data_manager.sync(ids_remove, ids_rename)
    validation_data_manager.sync(ids_remove_val, ids_rename_val)

    return ids_collect, ids_collect_val


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}!\nPlease make sure the file exists, and has all the necessary parameters.")
    with open(config_path, 'r') as f:
        return json.load(f)


def get_params():
    parser = argparse.ArgumentParser(description="Set parameters for the script.")
    
    # Paths
    parser.add_argument('--cache_folder', '-c', type=str)
    parser.add_argument('--dataset_path', '-d', type=str)
    parser.add_argument('--validation_dataset_path', '-vd', type=str)
    parser.add_argument('--teacher_models_folder', '-tm', type=str)
    parser.add_argument('--student_path', '-s', type=str)
    
    # Cache settings
    parser.add_argument('--max_cache_size_gb', '-maxgb', type=int)
    
    # Pipeline settings
    parser.add_argument('--ignore_model_type', type=bool)
    parser.add_argument('--rebase_dataset', type=bool)

    # General model settings
    parser.add_argument('--context_len', '-ctx', type=int)
    parser.add_argument('--save_sys_range', type=bool)
    parser.add_argument('--save_user_range', type=bool)
    parser.add_argument('--save_assistant_range', type=bool)
    parser.add_argument('--crop_distr_to_size', type=int)
    parser.add_argument('--enable_topK', type=bool)
    parser.add_argument('--save_topK', '-topk', type=int)
    parser.add_argument('--device', type=str)
    
    # Collection settings
    parser.add_argument('--num_inference_workers', '-niw', type=int)
    parser.add_argument('--reserve_vram', type=float, nargs='+')
    parser.add_argument('--encourage_eos', type=bool)
    
    # Training settings
    parser.add_argument('--num_epochs', '-ne', type=int)
    parser.add_argument('--num_warmup_steps', '-nws', type=int)
    parser.add_argument('--batch_size', '-bs', type=int)
    parser.add_argument('--grad_accum_batches', '-g', type=int)
    parser.add_argument('--grad_checkpointing', type=bool)
    parser.add_argument('--temperature', '-t', type=float)
    parser.add_argument('--lr', '-lr', type=float)
    parser.add_argument('--decay_start', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--lr_scheduler', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--data_order', type=str)
    parser.add_argument('--training_precision', type=str)
    parser.add_argument('--validate_every_n_epochs', type=float)
    parser.add_argument('--save_student_every_n_epochs', type=float)
    parser.add_argument('--num_gpu0_layers', type=int)
    parser.add_argument('--device_map', type=str)
    parser.add_argument('--max_memory', type=str)
    parser.add_argument('--multi_gpu', type=bool)
    parser.add_argument('--save_final_state', type=bool)
    parser.add_argument('--wandb_comment', '-wdb', type=str)
    
    # Student settings
    parser.add_argument('--freeze_layers', '-fl', type=str, nargs='+')
    parser.add_argument('--add_bos', type=bool)
    parser.add_argument('--prompt_format', type=json.loads)
    
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = load_config(config_path)

    params = {key: getattr(args, key) if getattr(args, key) is not None else config[key] for key in config.keys()}

    params['max_memory'] = {int(key): value for key, value in params['max_memory'].items() if key.lower() != 'cpu'}

    return params


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
    encourage_eos = params['encourage_eos']

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
    wandb_comment = params['wandb_comment'] # Comment for wandb: {comment} ModelName lr(lr) (Date/Time)

    # Student settings
    freeze_layers = params['freeze_layers']
    add_bos = params['add_bos']
    prompt_format = params['prompt_format']


    # Initialization
    multiprocessing.set_start_method('spawn', force=True)
    paths = Paths(cache_folder)
    teachers = get_teachers(teacher_models_folder)
    student = StudentModel(student_path, paths, add_bos, prompt_format, batch_size)

    ensure_compatibility(teachers, student)
    prepare_datasets(dataset_path, validation_dataset_path, teachers, student, context_len, save_sys_range, save_user_range, save_assistant_range, ignore_model_type)

    set_params(teachers, student, crop_distr_to_size, context_len, temperature, device, save_topK, enable_topK, encourage_eos)
    set_training_params(student, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_batches, training_precision, decay_start, multi_gpu, data_order,
                        validate_every_n_epochs, save_student_every_n_epochs, save_final_state, grad_checkpointing, freeze_layers, wandb_comment, alpha, device_map, max_memory, num_gpu0_layers)

    student.reorder_dataset()

    validation_data_manager = H5DataManager(paths.dataset_validation, device)
    time.sleep(2) # 2s sleep to allow the data manager to initialize, else it stalls for some reason
    data_manager = H5DataManager(paths.dataset, device)
    
    loop_ids, full_collect = calculate_loop_ids(student, max_cache_size_gb, enable_topK, save_topK)
    
    ids_collect, ids_collect_val = sync_datasets(validation_data_manager, data_manager, rebase_dataset, student)

    # Main logic

    ## Validation collection
    if ids_collect_val and not rebase_dataset:
        print(f"Collecting validation data for {len(ids_collect_val)} samples...")

        validation_data_manager.set_vocab_family(student.vocab_family)

        for teacher in teachers:
            teacher.process_chunk(reserve_vram, num_inference_workers, data_manager=validation_data_manager, ids_to_collect=ids_collect_val, validation=True)
    
    else:
        print("Using data on disk for validation...")
    
    
    ## Training collecting loop
    if ids_collect and not rebase_dataset:

        data_manager.set_vocab_family(student.vocab_family)

        if full_collect:
            print(f"Collecting data for {len(ids_collect)} samples...")
            for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False):
                teacher.process_chunk(reserve_vram, num_inference_workers, ids_to_collect=ids_collect, data_manager=data_manager)

            student.train_chunk(data_manager, validation_data_manager, full_collect)

        else:
            print(f"WARNING: The calculated size of the h5 dataset is too large for the specified cache size ({max_cache_size_gb}GB).")
            print("The data will be collected in chunks. This will erase all previous data in the main h5 dataset!")

            response = input("Do you wish to continue? (y/n): ")
            if response.lower() not in ["y", "ye", "yes", "1", "true", "t"]:
                print("Exiting...")
                validation_data_manager.close()
                data_manager.close()
                student.close()
                exit(0)

            for epoch in tqdm(range(num_epochs), desc="Epochs", smoothing=0.06, position=0):

                for chunk_ids in tqdm(loop_ids, desc="Chunks", smoothing=0.06, position=1, leave=False):
                    data_manager.purge_dataset()

                    for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=2, leave=False):
                        teacher.process_chunk(reserve_vram, num_inference_workers, ids_to_collect=chunk_ids, data_manager=data_manager)

                    student.train_chunk(data_manager, validation_data_manager, full_collect)

    else:
        print("Using data on disk for training...")

        student.train_chunk(data_manager, validation_data_manager, True)

    validation_data_manager.close()
    data_manager.close()
    student.close()

    print("Done!")

if __name__ == "__main__":
    main()
