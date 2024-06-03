from utils.dataset_utils import tokenize_dataset
from classes.teacher.model import TeacherModel
from classes.student.model import StudentModel
from classes.data_manager import H5DataManager
from classes.paths import Paths
from tqdm import tqdm
import multiprocessing
import nvidia_smi
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


def collect_or_load(student: StudentModel, teachers: list[TeacherModel], paths: Paths, rebase_dataset: bool) -> tuple[bool, bool]:

    def check_sha(path, current_sha):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return False

        disk_sha = {}
        with open(path, "r") as f:
            for line in f:
                json_line = json.loads(line)
                disk_sha.update(json_line)

        for id, sha in current_sha.items():
            if id not in disk_sha:
                if not rebase_dataset:
                    print(f"ID {id} not found in disk sha")
                return False
            if disk_sha[id] != sha:
                if not rebase_dataset:
                    print(f"Disk data ID {id} sha mismatch: {disk_sha[id]} vs {sha}")
                return False

        return True

    def check_teacher_names(path, teacher_names):
        if not os.path.exists(path):
            return False

        with open(path, "r") as f:
            disk_teacher_names = json.load(f)

        return disk_teacher_names == teacher_names

    def check_paths(paths, teacher_names, current_sha):
        distr_path = os.path.join(paths, "distributions.hdf5")
        names_path = os.path.join(paths, "teacher_names.json")
        sha_path = os.path.join(paths, "sha.jsonl")

        if not (os.path.exists(distr_path) and os.path.exists(names_path) and os.path.exists(sha_path)):
            return True

        if not check_teacher_names(names_path, teacher_names):
            return True

        if not check_sha(sha_path, current_sha):
            return True

        return False

    teacher_names = [teacher.model_name for teacher in teachers]
    current_sha = {f"{convo.origin_convo_id}": convo.content_sha for convo in student.dataset}
    current_sha_val = {f"{convo.origin_convo_id}": convo.content_sha for convo in student.validation_dataset}

    collect = check_paths(paths.dataset, teacher_names, current_sha)
    collect_val = check_paths(paths.dataset_validation, teacher_names, current_sha_val)

    return collect, collect_val


def save_dataset_metadata(path, teachers: list[TeacherModel], student: StudentModel, validation=False):
    if os.path.exists(os.path.join(path, "sha.jsonl")):
        os.remove(os.path.join(path, "sha.jsonl"))

    if os.path.exists(os.path.join(path, "teacher_names.json")):
        os.remove(os.path.join(path, "teacher_names.json"))

    with open(os.path.join(path, "sha.jsonl"), "w", encoding="utf-8") as f:

        if validation:
            for convo in student.validation_dataset:
                f.write(json.dumps({convo.origin_convo_id: convo.content_sha}, ensure_ascii=False) + "\n")
        else:
            for convo in student.dataset:
                f.write(json.dumps({convo.origin_convo_id: convo.content_sha}, ensure_ascii=False) + "\n")
            
    with open(os.path.join(path, "teacher_names.json"), "w", encoding="utf-8") as f:
        json.dump([teacher.model_name for teacher in teachers], f, indent=4, ensure_ascii=False)


def main():
    cache_folder = r"C:\Users\PC\Desktop\cache"
    max_cache_size_gb = 320

    # "/root/axo_clone/axolotl/data/random_samples_4k.jsonl"
    # "/root/axo_clone/Distill_Latest_Clone/train_test_small.jsonl"
    dataset_path = r"C:\Users\PC\Converted_random_samples_4k.jsonl"
    validation_dataset_path = r"C:\Users\PC\val_completion.jsonl"

    teacher_models_folder = r"C:\Users\PC\Desktop\teachers"
    student_path = r"C:\Users\PC\Desktop\TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"

    ignore_model_type = True # If True, will collect all data from teachers regardless if both, conversation and teacher are matching in being completion/instruct
    rebase_dataset = False # Only use if you know what you are doing. Will skip the check for conversation and teacher match to use data from disk

    # General model settings
    context_len = 2*1024
    save_sys_range = True
    save_user_range = True
    save_assistant_range = True
    crop_distr_to_size = 32000
    enable_topK = True
    save_topK = 300
    device = "cuda:1"

    # Collection settings
    num_inference_workers = 1 # 3 IS ABSOLUTE MAX IT CAN GO
    reserve_vram = [4, 0.5] # GB, reserving ordered as [cuda:0, cuda:1, cuda:2, cuda:3, ...]
    encourage_eos = False

    # Training settings
    num_epochs = 1
    num_warmup_steps = 50

    batch_size = 4
    grad_accum_batches = 1
    grad_checkpointing = False
    temperature = 1
    lr = 1e-6
    decay_start = 0.9 # wsd only
    alpha = 2 # weighted losses
    
    lr_scheduler = "cosine" # "wsd", "cosine", "linear", "constant"
    optimizer = "adamw" # "adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit", "sgd", "rmsprop", "rmsprop8bit", "rmsprop32bit", "adagrad"
    data_order = "sorted" # "shuffle", "native", "sorted"
    training_precision = "fp16" # "fp32", "fp16", "bf16", "4bit", "8bit"
    
    validate_every_n_epochs = 0.1
    save_student_every_n_epochs = 4

    num_gpu0_layers = 7 # Used only if device_map is set to "custom"
    device_map = "custom" # "balanced", "balanced_low_0", "custom"
    max_memory = {0: '20GB', 1: '20GB', 'cpu': '200GB'} # {0: '20GB', 1: '60GB', 2: '60GB', 3: '60GB', 'cpu': '200GB'}
    multi_gpu = True
    save_final_state = False
    wandb_comment = f"FFT" # Comment for wandb: {comment} ModelName lr(lr) (Date/Time)

    # Student settings
    freeze_layers = []
    add_bos = True
    prompt_format = {
        "SYS_START": "#System:\n",
        "USER_START": "#User:\n",
        "ASSISTANT_START": "#AI:\n",
        "SYS_END": "\n\n",
        "USER_END": "\n\n",
        "ASSISTANT_END": "\n\n",
    }


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
    
    loop_ids, full_collect = calculate_loop_ids(student, max_cache_size_gb, enable_topK, save_topK)

    collect, collect_val = collect_or_load(student, teachers, paths, rebase_dataset)

    validation_data_manager = H5DataManager(paths.dataset_validation, device)
    time.sleep(2)
    data_manager = H5DataManager(paths.dataset, device)

    # Main loop

    ## Validation collecting loop
    if collect_val and not rebase_dataset:
        print("Collecting validation data...")

        validation_data_manager.purge_dataset()

        for teacher in teachers:
            teacher.process_chunk(reserve_vram, num_inference_workers, data_manager=validation_data_manager, validation=True)

        save_dataset_metadata(paths.dataset_validation, teachers, student, validation=True)

    else:
        print("Using data on disk for validation...")
    
        if rebase_dataset:
            save_dataset_metadata(paths.dataset_validation, teachers, student, validation=True)
    
    
    ## Training collecting loop
    if collect and not rebase_dataset:
        print("Collecting all data..." if full_collect else "Collecting data in chunks...")
        
        data_manager.purge_dataset()

        if full_collect:
            for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False):
                teacher.process_chunk(reserve_vram, num_inference_workers, full_collect=True, data_manager=data_manager)

            save_dataset_metadata(paths.dataset, teachers, student)

            student.train_chunk(data_manager, validation_data_manager, full_collect)

        else:
            for epoch in tqdm(range(num_epochs), desc="Epochs", smoothing=0.06, position=0):

                for chunk_ids in tqdm(loop_ids, desc="Chunks", smoothing=0.06, position=1, leave=False):
                    data_manager.purge_dataset()

                    for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=2, leave=False):
                        teacher.process_chunk(reserve_vram, num_inference_workers, ids_to_collect=chunk_ids, data_manager=data_manager)

                    student.train_chunk(data_manager, validation_data_manager, full_collect)

    else:
        print("Using data on disk for training...")

        if rebase_dataset:
            save_dataset_metadata(paths.dataset, teachers, student)

        student.train_chunk(data_manager, validation_data_manager, True)

    validation_data_manager.close()
    data_manager.close()

    print("\nDone!")

    student.close()

if __name__ == "__main__":
    main()
