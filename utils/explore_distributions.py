from finetuning_utils import teacher_tensors_hander
import torch
from transformers import AutoTokenizer

distr_folder = r"F:\distilled\data-MNHTN-standardized-OpenPlatypus-Train\MythoMax-L2-Kimiko-v2-13b"
model_path = r"F:\MythoMax-L2-Kimiko-v2-13b"
conv_distr_gen = teacher_tensors_hander(distr_folder, "cuda:0", empty_convo_ids=[])
conv_1_distr = next(conv_distr_gen)
print(conv_1_distr.shape)
next_token_ids = torch.argmax(conv_1_distr, dim=1).tolist()
tokenizer = AutoTokenizer.from_pretrained(model_path)
next_tokens = tokenizer.convert_ids_to_tokens(next_token_ids)
print(next_tokens)