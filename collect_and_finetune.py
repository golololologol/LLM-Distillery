from peft import LoraConfig, get_peft_model, TaskType
from utils.dataset_utils import tokenize_dataset
from classes.teacher.model import TeacherModel
from classes.student.model import StudentModel
from classes.data_manager import H5DataManager
from classes.paths import Paths
from tqdm import tqdm
import multiprocessing
import time
import json
import os

    
def calculate_loop_stops(teacher: TeacherModel, max_cache_size_gb, enable_topK, save_topK):
    
    if enable_topK:
        kb_per_distr_val = (0.001953125 * 3) / 1.7 # storage of one FP16 value * 3 (for int32 index) / 1.7 (avg compression ratio)
        distr_size_kb = kb_per_distr_val * save_topK
    else:
        kb_per_distr_val = 0.001953125 / 1.55 # storage of one FP16 value / 1.55 (avg compression ratio)
        distr_size_kb = kb_per_distr_val * teacher.crop_to_size
    max_cache_size_kb = max_cache_size_gb * 1e6
    cache_size_kb = 0
    full_collect = False
    loop_stops = []
    
    for id, convo in enumerate(teacher.dataset):
        convo_kb = distr_size_kb * convo.len_content
        cache_size_kb += convo_kb
        
        if cache_size_kb >= max_cache_size_kb:
            loop_stops.append(id+1)
            cache_size_kb = 0
    
    if not loop_stops:
        full_collect = True

    if loop_stops and loop_stops[-1] != len(teacher.dataset):
        loop_stops.append(len(teacher.dataset))

    return loop_stops, full_collect


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


def set_training_params(student: StudentModel, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_batches, training_precision, decay_start, multi_gpu, data_order, validate_every_n_epochs,
                         custom_reduction, save_student_every_n_epochs, save_final_state, grad_checkpointing, freeze_layers, use_gradual_dora, target_modules, rank, alpha, perma_merge_weight, perma_merge_every_n_batches, target_all_linear):
    student.num_epochs = num_epochs
    student.num_warmup_steps = num_warmup_steps
    student.lr = lr
    student.lr_scheduler_name = lr_scheduler
    student.optimizer_name = optimizer
    student.num_grad_accum_batches = grad_accum_batches
    student.training_precision_name = training_precision
    student.decay_start = decay_start
    student.multi_gpu = multi_gpu
    student.data_order = data_order
    student.validation_every_steps = validate_every_n_epochs * student.dataset_len
    student.next_accum_step = grad_accum_batches * student.batch_size
    student.num_training_steps = num_epochs * student.dataset_len
    student.custom_reduction = custom_reduction
    student.save_every_steps = save_student_every_n_epochs * student.dataset_len
    student.next_save_step = save_student_every_n_epochs * student.dataset_len
    student.save_final_state = save_final_state
    student.grad_checkpointing = grad_checkpointing
    student.freeze_layers = freeze_layers

    # Dora
    student.use_dora = use_gradual_dora
    student.target_all_linear = target_all_linear
    student.target_modules = target_modules
    student.lora_rank = rank
    student.lora_alpha = alpha
    student.perma_merge_weight = perma_merge_weight
    student.perma_merge_every_batches = perma_merge_every_n_batches
    student.next_merge_step = perma_merge_every_n_batches * student.batch_size


def rewrite_teachers_param(teachers: list[TeacherModel], param_name, param_value):
    for teacher in teachers:
        setattr(teacher, param_name, param_value)


def sort_datasets_by_map(teachers: list[TeacherModel], sorting_map: dict[int, int], validation_sorting_map: dict[int, int]):
    for teacher in teachers:
        if not teacher.dataset_sorted:
           teacher.sort_dataset_by_map(sorting_map)
        
        if not teacher.validation_dataset_sorted:
            teacher.sort_dataset_by_map(validation_sorting_map, validation=True)


def collect_or_load(student: StudentModel, teachers: list[TeacherModel], paths: Paths) -> tuple[bool, bool]:

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
                print(f"ID {id} not found in disk sha")
                return False
            if disk_sha[id] != sha:
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
    max_cache_size_gb = 300

    # "/root/axo_clone/axolotl/data/random_samples_4k.jsonl"
    # "/root/axo_clone/Distill_Latest_Clone/train_test_small.jsonl"
    dataset_path = r"C:\Users\PC\Desktop\train_test_small.jsonl"
    validation_dataset_path = r"C:\Users\PC\Desktop\val_test.jsonl"

    teacher_models_folder = r"C:\Users\PC\Desktop\teachers"
    student_path = r"C:\Users\PC\Downloads\tiny-mistral-200M"

    ignore_model_type = True # If True, will collect all data from teachers regardless if both, conversation and teacher are matching in being completion/instruct

    # General model settings
    context_len = 2*1024
    save_sys_range = True
    save_user_range = True
    save_assistant_range = True
    crop_distr_to_size = 32000
    enable_topK = True
    save_topK = 200
    device = "cuda:1"
    reserve_vram = [6, 0.5] # GB

    # Collection settings
    encourage_eos = False

    # Training settings
    num_epochs = 1
    num_warmup_steps = 200
    ## Dora
    use_gradual_dora = False
    target_all_linear = True
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] # Only used if target_all_linear is False
    rank = 32
    alpha = 32
    perma_merge_weight = 0.1 # 0.1 = 10% gets merged every merging step
    perma_merge_every_n_batches = 1

    batch_size = 4
    grad_accum_batches = 1
    grad_checkpointing = False
    temperature = 1
    lr = 5e-5
    decay_start = 0.9 # wsd only

    lr_scheduler = "wsd" # "wsd", "cosine", "linear", "constant"
    optimizer = "adamw" # "adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit", "sgd", "rmsprop32bit"
    data_order = "shuffle" # "shuffle", "native", "sorted"
    training_precision = "bf16" # "fp32", "fp16", "bf16", "4bit", "8bit"
    
    validate_every_n_epochs = 0.5
    save_student_every_n_epochs = 1

    multi_gpu = True
    custom_reduction = True
    save_final_state = True

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
    set_training_params(student, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_batches, training_precision, decay_start, multi_gpu, data_order, validate_every_n_epochs, 
                        custom_reduction, save_student_every_n_epochs, save_final_state, grad_checkpointing, freeze_layers, use_gradual_dora, target_modules, rank, alpha, perma_merge_weight, perma_merge_every_n_batches, target_all_linear)

    sorting_map, validation_sorting_map = teachers[0].sort_dataset_by_len()
    sort_datasets_by_map(teachers, sorting_map, validation_sorting_map)

    loop_stops, full_collect = calculate_loop_stops(teachers[0], max_cache_size_gb, enable_topK, save_topK)

    collect, collect_val = collect_or_load(student, teachers, paths)

    # Main loop

    ## Validation collecting loop
    
    validation_data_manager = H5DataManager(paths.dataset_validation, device)
    time.sleep(2)
    data_manager = H5DataManager(paths.dataset, device)

    if collect_val:
        print("Collecting validation data...")

        validation_data_manager.purge_dataset()

        for teacher in teachers:
            teacher.process_chunk(reserve_vram, full_collect=True, data_manager=validation_data_manager, validation=True)

        save_dataset_metadata(paths.dataset_validation, teachers, student, validation=True)

    else:
        print("Using data on disk for validation...")
    
    ## Training collecting loop
    if collect:
        print("Collecting all data..." if full_collect else "Collecting data in chunks...")

        data_manager.purge_dataset()

        if full_collect:
            for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False):
                teacher.process_chunk(reserve_vram, full_collect=True, data_manager=data_manager)

            save_dataset_metadata(paths.dataset, teachers, student)

            student.train_chunk(data_manager, validation_data_manager, full_collect)

        else:
            for epoch in tqdm(range(num_epochs), desc="Epochs", smoothing=0.06, position=0):

                for stop in tqdm(loop_stops, desc="Chunks", smoothing=0.06, position=1, leave=False):
                    data_manager.purge_dataset()

                    for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=2, leave=False):
                        teacher.process_chunk(reserve_vram, next_stop_id=stop, data_manager=data_manager)

                    student.train_chunk(data_manager, validation_data_manager, full_collect)

                for teacher in teachers:
                    teacher.new_epoch()
    else:
        print("Using data on disk for training...")
        student.train_chunk(data_manager, validation_data_manager, True)

    validation_data_manager.close()
    data_manager.close()

    print("\nDone!")

    student.close()

if __name__ == "__main__":
    main()
