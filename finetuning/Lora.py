import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from tqdm import tqdm
from utils.finetuning_utils import calculate_kl_divergence, teacher_tensors_hander

def lora_finetune(parameters):
    pass