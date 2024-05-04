import os
from tqdm import tqdm
from utils.dataset_utils import tokenize_dataset, H5DataManager
from utils.classes import TeacherModel, StudentModel, Paths
    
def calculate_loop_stops(teacher: TeacherModel, max_cache_size_gb):
    kb_per_distr_val = 0.001953125 # storage of one FP16 value
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
        
def prepare_datasets(dataset_path, validation_dataset_path, teachers: list[TeacherModel], student: StudentModel, context_len, save_sys_range, save_user_range, save_assistant_range):
    print("Preparing datasets for the student...")
    student.dataset, relevant_ids = tokenize_dataset(
        dataset_path, student.model_path, student.prompt_format, context_len,
        save_sys_range, save_user_range, save_assistant_range, add_bos=student.add_bos)

    student.validation_dataset, relevant_ids_val = tokenize_dataset(
        validation_dataset_path, student.model_path, student.prompt_format, context_len,
        save_sys_range, save_user_range, save_assistant_range, add_bos=student.add_bos)

    teachers_ids = set()
    teachers_ids_val = set()

    for teacher in teachers:
        print(f"Preparing datasets for {teacher.model_name}...")
        teacher.dataset, teacher_ids = tokenize_dataset(
            dataset_path, teacher.model_path, teacher.prompt_format, context_len,
            save_sys_range, save_user_range, save_assistant_range, add_bos=teacher.add_bos)

        teacher.validation_dataset, teacher_ids_val = tokenize_dataset(
            validation_dataset_path, teacher.model_path,teacher.prompt_format,context_len,
            save_sys_range, save_user_range, save_assistant_range, add_bos=teacher.add_bos)
        
        teachers_ids.update(teacher_ids)
        teachers_ids_val.update(teacher_ids_val)

    common_ids = relevant_ids.intersection(teachers_ids)
    common_ids_val = relevant_ids_val.intersection(teachers_ids_val)

    student.dataset = [convo for convo in student.dataset if convo.origin_convo_id in common_ids]
    student.dataset_len = len(student.dataset)
    student.validation_dataset = [convo for convo in student.validation_dataset if convo.origin_convo_id in common_ids_val]
    student.validation_dataset_len = len(student.validation_dataset)

    for teacher in teachers:
        teacher.dataset = [convo for convo in teacher.dataset if convo.origin_convo_id in common_ids]
        teacher.dataset_len = len(teacher.dataset)
        teacher.validation_dataset = [convo for convo in teacher.validation_dataset if convo.origin_convo_id in common_ids_val]
        teacher.validation_dataset_len = len(teacher.validation_dataset)
    
def set_params(teachers: list[TeacherModel], student: StudentModel, crop_to_size: int, context_len: int, temperature: float, device: str):
    for teacher in teachers:
        teacher.crop_to_size = crop_to_size
        teacher.context_len = context_len
        teacher.temperature = temperature
        teacher.student_eos_id = student.special_tokens["eos_id"]
        teacher.device = device

    student.crop_to_size = crop_to_size
    student.context_len = context_len
    student.temperature = temperature
    student.device = device

def set_training_params(student: StudentModel, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_steps, training_precision,
                         decay_start, multi_gpu, data_order, validate_every_n_epochs, custom_reduction, save_student_every_n_epochs):
    student.num_epochs = num_epochs
    student.num_warmup_steps = num_warmup_steps
    student.lr = lr
    student.lr_scheduler_name = lr_scheduler
    student.optimizer_name = optimizer
    student.grad_accum_steps = grad_accum_steps
    student.training_precision_name = training_precision
    student.decay_start = decay_start
    student.multi_gpu = multi_gpu
    student.data_order = data_order
    student.validation_per_steps = validate_every_n_epochs * student.validation_dataset_len
    student.next_accum_step = grad_accum_steps
    student.num_training_steps = num_epochs * student.dataset_len
    student.custom_reduction = custom_reduction
    student.save_every_steps = save_student_every_n_epochs * student.dataset_len
    student.next_save_step = save_student_every_n_epochs * student.dataset_len

def rewrite_teachers_param(teachers: list[TeacherModel], param_name, param_value):
    for teacher in teachers:
        setattr(teacher, param_name, param_value)

def sort_datasets_by_map(teachers: list[TeacherModel], sorting_map: dict[int, int], validation_sorting_map: dict[int, int]):
    for teacher in teachers:
        if not teacher.dataset_sorted:
           teacher.sort_dataset_by_map(sorting_map)
        
        if not teacher.validation_dataset_sorted:
            teacher.sort_dataset_by_map(validation_sorting_map, validation=True)

def main():
    cache_folder = r"C:\Users\PC\Desktop\cache"
    max_cache_size_gb = 200

    dataset_path = r"C:\Users\PC\Desktop\train_test.jsonl"
    validation_dataset_path = r"C:\Users\PC\Desktop\val_test.jsonl"

    teacher_models_folder = r"C:\Users\PC\Desktop\teachers"
    student_path = r"C:\Users\PC\Desktop\TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"

    # General model settings
    context_len = 2*1024
    save_sys_range = False
    save_user_range = False
    save_assistant_range = True
    crop_distr_to_size = 32000
    device = "cuda:1"
    reserve_vram = [5, 0.2] # GB

    # Training settings
    num_epochs = 8
    num_warmup_steps = 200
    temperature = 2
    lr = 2e-6
    lr_scheduler = "wsd" # "wsd", "cosine", "linear", "constant"
    optimizer = "adamw32bit" # "adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit", "sgd", "rmsprop32bit"
    data_order = "shuffle" # "shuffle", "native", "sorted"
    grad_accum_steps = 1
    training_precision = "fp16" # "fp32", "fp16", "bf16", "4bit", "8bit"
    multi_gpu = True
    decay_start = 0.9 # wsd only
    validate_every_n_epochs = 0.5
    save_student_every_n_epochs = 1
    custom_reduction = True

    # Initialization
    paths = Paths(cache_folder, clean_start=True)
    teachers = get_teachers(teacher_models_folder)
    student = StudentModel(student_path, paths)
    ensure_compatibility(teachers, student)
    prepare_datasets(dataset_path, validation_dataset_path, teachers, student, context_len, save_sys_range, save_user_range, save_assistant_range)

    set_params(teachers, student, crop_distr_to_size, context_len, temperature, device)
    set_training_params(student, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_steps, training_precision, decay_start,
                         multi_gpu, data_order, validate_every_n_epochs, custom_reduction, save_student_every_n_epochs)

    sorting_map, validation_sorting_map = teachers[0].sort_dataset_by_len()
    sort_datasets_by_map(teachers, sorting_map, validation_sorting_map)

    loop_stops, full_collect = calculate_loop_stops(teachers[0], max_cache_size_gb)
    print(f"Loop stops: {loop_stops}")
    # Main loop

    ## Validation collecting loop
    print("Processing validation data...")
    validation_data_manager = H5DataManager(paths.dataset_validation, device)

    for teacher in teachers:
        teacher.process_chunk(reserve_vram, full_collect=True, data_manager=validation_data_manager, validation=True)

    ## Training collecting loop
    print("Processing all data..." if full_collect else "Processing data in chunks...")
    data_manager = H5DataManager(paths.dataset, device)

    if full_collect:
        for teacher in tqdm(teachers, desc="Teachers", smoothing=0.06, position=0, leave=False):
            teacher.process_chunk(reserve_vram, full_collect=True, data_manager=data_manager)
        
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

    print("\nDone!\n")

    input("Press Enter to close tensorboard and program...")

    student.close()
    validation_data_manager.close()
    data_manager.close()

if __name__ == "__main__":
    main()
