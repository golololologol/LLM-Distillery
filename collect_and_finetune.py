from classes.args import get_config, PipelineConfig
from utils.dataset_utils import tokenize_dataset
from classes.teacher.model import TeacherModel
from classes.student.model import StudentModel
from classes.data_manager import H5DataManager
from joblib import Parallel, delayed
from classes.paths import Paths
from tqdm import tqdm
import multiprocessing
import nvidia_smi
import signal
import math
import time
import json
import os
import wandb
import sys

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


def get_teachers(config: PipelineConfig) -> list[TeacherModel]:
    """
    Loads teacher models if use_teachers is True, else returns an empty list.
    """
    teachers = []
    models_folder = config.teacher_models_folder

    if not config.use_teachers:
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
    Ensures that teachers and student models are all compatible for distillation.
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
        

def prepare_datasets(dataset_path, data_manager: H5DataManager, validation_dataset_path, validation_data_manager: H5DataManager, teachers: list[TeacherModel], student: StudentModel, config: PipelineConfig):
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


def set_training_params(student: StudentModel, num_epochs, num_warmup_steps, lr, adam_betas, adam_decay, lr_scheduler, optimizer, grad_accum_batches, training_precision, lr_decay_start, multi_gpu, data_order, validate_every_n_epochs, 
                        save_student_every_n_epochs, save_final_state, grad_checkpointing, freeze_layers, wandb_comment, wandb_project, alpha, device_map, max_memory, num_gpu0_layers, use_flash_attn_2):
    """
    Configures and schedules all essential training hyperparameters.
    """
    student.num_epochs = num_epochs
    student.eff_batch_size = grad_accum_batches * student.batch_size
    student.num_warmup_steps = math.ceil(num_warmup_steps / student.eff_batch_size)
    student.total_training_steps = num_epochs * student.dataset_len
    student.lr = lr
    student.adam_betas = adam_betas
    student.adam_decay = adam_decay
    student.lr_scheduler_name = lr_scheduler.lower()
    student.optimizer_name = optimizer.lower()
    student.grad_accum = grad_accum_batches
    student.num_grad_accum_batches = math.ceil(student.total_training_steps / student.eff_batch_size)
    student.training_precision_name = training_precision.lower()
    student.lr_decay_start = lr_decay_start
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
    student.wandb_project = wandb_project
    student.alpha = alpha
    student.device_map_name = device_map
    student.max_memory = max_memory
    student.num_gpu0_layers = num_gpu0_layers


def calculate_sync(student_dataset: list, data_manager: H5DataManager, student_vocab_family: str):
    """
    Calculates the IDs to collect, rename, and remove based on the current SHAs and the disk SHAs.
    
    This function is used to synchronize the datasets between the current text dataset and the logit dataset on disk.
    
    Args:
        student_dataset (list): The dataset to be synchronized.
        data_manager (H5DataManager): The data manager for the dataset.
        student_vocab_family (str): The vocabulary family of the student model.
        

    Returns:
        tuple: with three lists:
            - IDs to collect from the main dataset.
            - IDs to rename in the main dataset.
            - IDs to remove from the main dataset.
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


def sync_datasets(validation_data_manager: H5DataManager, data_manager: H5DataManager, rebase: bool, student: StudentModel, use_teachers: bool):
    """
    Synchronize main and validation datasets.

    If rebase is True, update both data managers with the student's current SHAs and vocab family,
    and return empty lists. If use_teachers is False, no sync is performed and empty lists are returned.
    Otherwise, perform teacher-led synchronization by computing and applying SHA changes, and return
    lists of IDs to collect for the main and validation datasets.

    Args:
        validation_data_manager (H5DataManager): Manager for the validation dataset.
        data_manager (H5DataManager): Manager for the main dataset.
        rebase (bool): If True, reset SHAs and vocabulary families.
        student (StudentModel): Student model containing datasets and vocab_family.
        use_teachers (bool): If True, perform teacher-based synchronization.

    Returns:
        tuple: Two lists containing IDs to collect from the main and validation datasets, respectively.
    """
    
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


def check_topk(data_manager: H5DataManager, enable_topk, save_topk: int, check_validation=False):
    """Checks if the topk value in the dataset matches the one in the config.

    Args:
        data_manager (H5DataManager): data_manager object to check the topk value.
        enable_topk (_type_): argument to enable or disable topk sampling.
        save_topk (int): topk value to save.
        validation (bool, optional): whether to check the validation dataset, else the main one. Defaults to False.

    Returns:
        int: topk value to use.
    """
    if not enable_topk:
        return save_topk
    
    dataset_topk = data_manager.get_topk()

    if dataset_topk is None:
        data_manager.set_topk(save_topk)
        return save_topk
    
    text_insert = "validation" if check_validation else "main"

    if dataset_topk != save_topk:
        print(f"\nTopK mismatch between the {text_insert} h5 dataset, and your config!\n{text_insert.upper()} dataset has TopK={dataset_topk}, while you set TopK={save_topk}.")
        response = input("Do you wish to continue with the dataset's TopK?\nIf not, the program will exit to allow you to change the config. (y/n): ")

        if response.lower() not in ["y", "ye", "yes", "1", "true", "t"]:
            handle_termination(None, None)
        
        return dataset_topk

    return save_topk
        


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
    config = get_config()

    # Initialization
    global data_manager, validation_data_manager, teachers, student
    
    if not wandb.api.api_key:
        print("Not logged in to wandb! (`wandb.api.api_key` is empty)\nPlease type `wandb login` in the terminal before next launch of the pipeline!\nExiting...")
        handle_termination(None, None)

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)

    multiprocessing.set_start_method('spawn', force=True)
    paths = Paths(config.cache_folder)
    teachers = get_teachers(config)
    student = StudentModel(config)# REDO

    print("Launching data managers...")
    data_manager = H5DataManager(paths.dataset, config.device, manager_name="main")
    validation_data_manager = H5DataManager(paths.dataset_validation, config.device, manager_name="validation")
    
    ensure_compatibility(teachers, student, config.use_teachers)
    prepare_datasets(data_manager, validation_data_manager, teachers, student, config)# REDO
    
    student.reorder_dataset()
    
    loop_ids, full_collect = calculate_loop_ids(student, config)# REDO
    
    ids_collect, ids_collect_val = sync_datasets(validation_data_manager, data_manager, student) # REDO

    # Main logic

    ## Validation collection
    if ids_collect_val and not config.rebase_dataset:
        print(f"Collecting validation data for {len(ids_collect_val)} samples...")

        topk_to_use = check_topk(validation_data_manager, config.enable_topK, config.save_topK, validation=True)
        update_teachers_param(teachers, "topK", topk_to_use)

        validation_data_manager.set_vocab_family(student.vocab_family)

        for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False, disable=len(teachers) == 1):
            teacher.process_chunk(config, data_manager=validation_data_manager, ids_to_collect=ids_collect_val, validation=True)# REDO
    
    else:
        print("Using data on disk for validation...")
    
    
    ## Training collection and finetuning
    if ids_collect and not config.rebase_dataset:

        data_manager.set_vocab_family(student.vocab_family)

        if full_collect:
            print(f"Collecting data for {len(ids_collect)} samples...")

            topk_to_use = check_topk(data_manager, config.enable_topK, config.save_topK)
            update_teachers_param(teachers, "topK", topk_to_use)

            for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False, disable=len(teachers) == 1):
                teacher.process_chunk(config, ids_to_collect=ids_collect, data_manager=data_manager)# REDO

            student.train_chunk(data_manager, validation_data_manager, full_collect)

        else:
            print(f"WARNING: The calculated size of the h5 dataset is too large for the specified cache size ({config.max_cache_size_gb}GB).")
            print("The data will be collected and trained on in chunks.")

            if data_manager.has_data():
                print("This will erase all previous data in the main h5 dataset!")
                response = input("Do you wish to continue? (y/n): ")

                if response.lower() not in ["y", "ye", "yes", "1", "true", "t"]:
                    handle_termination(None, None)

            data_manager.set_topk(config.save_topK)

            for epoch in tqdm(range(config.num_epochs), desc="Epochs", smoothing=0.06, position=0):

                for chunk_ids in tqdm(loop_ids, desc="Chunks", smoothing=0.06, position=1, leave=False):
                    data_manager.purge_dataset(ask_confirmation=False)

                    for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=2, leave=False):
                        teacher.process_chunk(config, ids_to_collect=chunk_ids, data_manager=data_manager)# REDO

                    student.train_chunk(data_manager, validation_data_manager, full_collect)

    else:
        print("Using data on disk for training...")

        student.train_chunk(data_manager, validation_data_manager, True)

    print("Done!")

    handle_termination(None, None)

if __name__ == "__main__":
    main()
