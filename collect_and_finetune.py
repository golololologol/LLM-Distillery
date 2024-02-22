import os
import json
from utils.dataset_utils import is_model_safetensors, load_prompt_format, save_prompt_format, tokenize_dataset, get_vocab_family, get_special_tokens
from utils.classes import TeacherModel, StudentModel
from utils.convert_to_safetensor import convert_model

def load_config(model_path):
    config_path = os.path.join(model_path, "teacher_config.json")
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_config(config, model_path):
    config_path = os.path.join(model_path, "teacher_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def input_prompt_format():
    prompt_format = {
        'SYS_START': "### System:\n",
        'USER_START': "### User:\n",
        'ASSISTANT_START': "### Assistant:\n",
        'SYS_END': '\n',
        'USER_END': '\n',
        'ASSISTANT_END': '\n'
    }

    keys = list(prompt_format.keys())
    i = 0

    while i < len(keys):
        key = keys[i]
        default_value = prompt_format[key].encode('unicode_escape').decode()
        value = input(f"{key} (default: {default_value}): ")
        
        if value == "<":
            i = max(0, i - 1)
        else:
            prompt_format[key] = value
            i += 1
            
    return prompt_format

def input_config():
    config = {
        'batch_size': 4,
        'add_bos': True,
        'context_chunk_size': 1024
    }
    keys = list(config.keys())
    i = 0

    while i < len(keys):
        key = keys[i]
        default_value = str(config[key])
        value = input(f"{key} (default: {default_value}): ")
        
        if value == "<":
            i = max(0, i - 1)
        else:
            config[key] = value
            i += 1

    return config
    
def get_teachers(models_folder):
    teachers = []
    model_names = os.listdir(models_folder)

    if not model_names:
        print("No models found in the teacher models folder")
        exit(0)

    for model_name in model_names:
        model_path = os.path.join(models_folder, model_name)
        if not is_model_safetensors(model_path):
            model_path = convert_model(model_path)

        pf = load_prompt_format(model_path)
        if pf is None:
            print(f"{model_name} has no prompt format")
            pf = input_prompt_format()
            save_prompt_format(pf, model_path)

        config = load_config(model_path)
        if config is None:
            print(f"{model_name} has no config")
            config = input_config()
            save_config(config, model_path)
        
        vocab_family = get_vocab_family(model_path)
        special_tokens = get_special_tokens(vocab_family)
        teachers.append(TeacherModel(model_path, model_name, pf, config, vocab_family, special_tokens))
    
    return teachers
            
def ensure_compatability(teachers: list[TeacherModel], student: StudentModel):
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
    
def set_params(teachers: list[TeacherModel], student: StudentModel, crop_to_size):
    for teacher in teachers:
        teacher.crop_to_size = crop_to_size

    student.crop_to_size = crop_to_size
        

        


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

