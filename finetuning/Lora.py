import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from tqdm import tqdm
from utils.finetuning_utils import calculate_kl_divergence

def lora_finetune(parameters):
    pass