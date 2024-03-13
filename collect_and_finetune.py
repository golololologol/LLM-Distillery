import os
from tqdm import tqdm
from utils.dataset_utils import tokenize_dataset, H5DataManager
from utils.classes import TeacherModel, StudentModel, Paths
    
def calculate_loop_stops(teacher: TeacherModel, max_cache_size_gb):
    kb_per_distr_val = 0.001953125 # storage of one FP16 value
    distr_size_kb = kb_per_distr_val * teacher.crop_to_size
    max_cache_size_kb = max_cache_size_gb * 1e6
    cache_size_kb = 0
    loop_stops = []
    
    for id, convo in enumerate(teacher.dataset):
        convo_kb = distr_size_kb * convo.len_content
        cache_size_kb += convo_kb
        
        if cache_size_kb >= max_cache_size_kb:
            loop_stops.append(id+1)
            cache_size_kb = 0
    
    if not loop_stops or loop_stops[-1] != len(teacher.dataset):
        loop_stops.append(len(teacher.dataset))
    
    return loop_stops

def get_teachers(models_folder) -> list[TeacherModel]:
    teachers = []
    model_paths = [os.path.join(models_folder, model_name) for model_name in os.listdir(models_folder)]

    if not model_paths:
        print("No models found in the teacher models folder")
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
        
def prepare_teacher_datasets(dataset_path, validation_dataset_path, ppl_dataset_path, teachers: list[TeacherModel], context_len, save_sys_range, save_user_range, save_assistant_range):
    for teacher in teachers:
        print(f"Preparing datasets for {teacher.model_name}...")
        teacher.dataset = tokenize_dataset(
            dataset_path, teacher.model_path, teacher.prompt_format, context_len,
            save_sys_range, save_user_range, save_assistant_range, add_bos=teacher.add_bos)

        teacher.validation_dataset = tokenize_dataset(
            validation_dataset_path, teacher.model_path,teacher.prompt_format,context_len,
            save_sys_range, save_user_range, save_assistant_range, add_bos=teacher.add_bos)
        
        teacher.ppl_dataset = tokenize_dataset(
            ppl_dataset_path, teacher.model_path,teacher.prompt_format,context_len,
            save_sys_range, save_user_range, save_assistant_range, add_bos=teacher.add_bos)
    
def set_params(teachers: list[TeacherModel], student: StudentModel, crop_to_size, context_len, temperature):
    for teacher in teachers:
        teacher.crop_to_size = crop_to_size
        teacher.context_len = context_len
        teacher.temperature = temperature
        teacher.dataset_len = len(teacher.dataset)

    student.crop_to_size = crop_to_size
    student.context_len = context_len
    student.temperature = temperature

def rewrite_teachers_param(teachers: list[TeacherModel], param_name, param_value):
    for teacher in teachers:
        setattr(teacher, param_name, param_value)

def sort_datasets_by_map(teachers: list[TeacherModel], sorting_map):
    for teacher in teachers:
        teacher.sort_dataset_by_map(sorting_map)

def main():
    cache_folder = r"F:\cache"
    max_cache_size_gb = 200

    dataset_path = r"F:\ppl_test_dataset.jsonl"
    teacher_models_folder = r"F:\teacher_models"
    student_path = r"C:\Users\gololo\Desktop\TinyLlama-1.1B-intermediate-step-1431k-3T"

    ppl_dataset_path = r"F:\ppl_test_dataset.jsonl"
    validation_dataset_path = r"F:\ppl_test_dataset.jsonl"

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
    lr_scheduler = "linear"
    optimizer = "adamw"
    grad_accum_steps = 1
    decay_start = 0.9

    # Collecting logic
    teachers = get_teachers(teacher_models_folder)
    student = StudentModel(student_path)
    prepare_teacher_datasets(dataset_path, validation_dataset_path, ppl_dataset_path, teachers, context_len, save_sys_range, save_user_range, save_assistant_range)
    set_params(teachers, student, crop_distr_to_size, context_len, temperature)
    ensure_compatibility(teachers, student)
    
    # Initialization
    sorting_map = teachers[0].sort_dataset_by_len()
    sort_datasets_by_map(teachers, sorting_map)
    loop_stops = calculate_loop_stops(teachers[0], max_cache_size_gb)
    paths = Paths(cache_folder, clean_start=True)
   
    data_manager = H5DataManager(paths.dataset, device)

    # Collecting loop
    for stop_id in tqdm(loop_stops, desc="Chunks", position=0, smoothing=0.06, leave=False):
        rewrite_teachers_param(teachers, "next_stop_id", stop_id)

        for teacher in tqdm(teachers, desc="Teachers", position=1, smoothing=0.06, leave=False):
            assert isinstance(teacher, TeacherModel)

            teacher.load_model(reserve_vram_gb=reserve_vram)

            total_convos = stop_id - teacher.convo_id
            pbar_convos = tqdm(total=total_convos, desc="Convos", position=2, smoothing=0.06, leave=False)

            while teacher.convo_id < stop_id:
                batch_content_logprobs, batch_size = teacher.get_batch_content_logprobs()
                data_manager.write_batch(batch_content_logprobs)
                pbar_convos.update(batch_size)

            pbar_convos.close()
            teacher.unload_model()
        

            

                    
                
            


     



        



if __name__ == "__main__":
    main()
