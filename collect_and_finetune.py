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
    
    if not loop_stops or loop_stops[-1] < len(teacher.dataset):
        full_collect = True

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
            raise ValueError(f"Student has {key}={getattr(student, key)} while the teachers have {key}={value}")
        
def prepare_teacher_datasets(dataset_path, validation_dataset_path, teachers: list[TeacherModel], context_len, save_sys_range, save_user_range, save_assistant_range):
    for teacher in teachers:
        print(f"Preparing datasets for {teacher.model_name}...")
        teacher.dataset = tokenize_dataset(
            dataset_path, teacher.model_path, teacher.prompt_format, context_len,
            save_sys_range, save_user_range, save_assistant_range, add_bos=teacher.add_bos)
        teacher.dataset_len = len(teacher.dataset)

        teacher.validation_dataset = tokenize_dataset(
            validation_dataset_path, teacher.model_path,teacher.prompt_format,context_len,
            save_sys_range, save_user_range, save_assistant_range, add_bos=teacher.add_bos)
        teacher.validation_dataset_len = len(teacher.validation_dataset)
    
def set_params(teachers: list[TeacherModel], student: StudentModel, crop_to_size: int, context_len: int, temperature: float):
    for teacher in teachers:
        teacher.crop_to_size = crop_to_size
        teacher.context_len = context_len
        teacher.temperature = temperature

    student.crop_to_size = crop_to_size
    student.context_len = context_len
    student.temperature = temperature

def set_training_params(student: StudentModel, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_steps, training_precision, decay_start, multi_gpu):
    student.num_epochs = num_epochs
    student.num_warmup_steps = num_warmup_steps
    student.lr = lr
    student.lr_scheduler_name = lr_scheduler
    student.optimizer_name = optimizer
    student.grad_accum_steps = grad_accum_steps
    student.training_precision_name = training_precision
    student.decay_start = decay_start
    student.multi_gpu = multi_gpu

def rewrite_teachers_param(teachers: list[TeacherModel], param_name, param_value):
    for teacher in teachers:
        setattr(teacher, param_name, param_value)

def main():
    cache_folder = r"F:\cache"
    max_cache_size_gb = 200

    dataset_path = r"F:\ppl_test_dataset.jsonl"
    validation_dataset_path = r"F:\smol_test.jsonl"

    teacher_models_folder = r"F:\teacher_models"
    student_path = r"C:\Users\gololo\Desktop\TinyLlama-1.1B-intermediate-step-1431k-3T"

    # General model settings
    context_len = 2*1024
    save_sys_range = False
    save_user_range = True
    save_assistant_range = True
    crop_distr_to_size = 32000
    device = "cuda:1"
    reserve_vram = [1.3, 0.2] # GB

    # Training settings
    num_epochs = 1
    num_warmup_steps = 1000
    temperature = 3
    lr = 1e-4
    lr_scheduler = "wsd"
    optimizer = "adamw"
    grad_accum_steps = 1
    training_precision = "fp16" # "fp32", "fp16", "bf16", "4bit"
    multi_gpu = False
    decay_start = 0.9

    # Initialization
    paths = Paths(cache_folder, clean_start=True)
    teachers = get_teachers(teacher_models_folder)
    student = StudentModel(student_path, paths)
    ensure_compatibility(teachers, student)
    prepare_teacher_datasets(dataset_path, validation_dataset_path, teachers, context_len, save_sys_range, save_user_range, save_assistant_range)
    set_params(teachers, student, crop_distr_to_size, context_len, temperature)
    set_training_params(student, num_epochs, num_warmup_steps, lr, lr_scheduler, optimizer, grad_accum_steps, training_precision, decay_start, multi_gpu)
    loop_stops, full_collect = calculate_loop_stops(teachers[0], max_cache_size_gb)
    
    # Main loop

    ## Validation collecting loop
    print("Processing validation data...")
    validation_data_manager = H5DataManager(paths.dataset_validation, device)

    for teacher in teachers:
        teacher.process_chunk(reserve_vram, full_collect=True, data_manager=validation_data_manager, validation=True)

    validation_data_manager.close()

    ## Training collecting loop
    print("Processing all data..." if full_collect else "Processing data in chunks...")
    data_manager = H5DataManager(paths.dataset, device)

    if full_collect:
        for teacher in teachers:
            teacher.process_chunk(reserve_vram, full_collect=True, data_manager=data_manager)
        
    else:
        pbar = tqdm(total=len(loop_stops), desc="Stops", smoothing=0.06)
        for stop in loop_stops:
            for teacher in teachers:
                print(f"Collecting data for {teacher.model_name}...")
                teacher.process_chunk(reserve_vram, next_stop_id=stop)

            student.train_chunk()

            

                    
                
            


     



        



if __name__ == "__main__":
    main()
