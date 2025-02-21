from multiprocessing.managers import SharedMemoryManager
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoConfig
import math
import torch.nn.functional as F
import joblib
import json
import nvidia_smi
import time
import codecs
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2
from tqdm import tqdm
from collections import defaultdict
import subprocess
import os
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import ShareableList

# init cuda
if not torch.cuda.is_initialized():
    torch.cuda.init()
#from dataset_utils import H5Reader, H5Writer

#d1 = np.random.random(size = (10,2))
#d2 = torch.tensor(([1]), dtype=torch.float16)
#q = queue.Queue()
#q.put(d1)
#q.put(d2)
#with h5py.File('test.h5', 'a') as hf:
    #count = 0
    #while not q.empty():
        #data = q.get()
        # make a new dataset if it doesn't exist yet
        #if f'data_{count}' not in hf:
            #hf.create_dataset(f'data_{count}', data=data)
        #else:
            # otherwise, rewrite the existing dataset
            #hf.__delitem__(f'data_{count}')
            #hf.create_dataset(f'data_{count}', data=data)
        #count += 1

#with h5py.File('test.h5', 'r') as hf:
    #print(hf.keys())
    #print(torch.tensor(hf['data_0']))
    #print(torch.tensor(hf['data_1']))


#for i in range(1):
    #tensor = reader.read_next()
    #print(f"Tensor {i} size: {tensor.size()}")
    #print(tensor[0].sum())
#temp = 2
#student_logits = torch.tensor(([0.0004, 10.1, 1.2], [1.5, 2.3, 3.4]))
#teacher_probs = nn.functional.softmax(torch.tensor(([0.0004, 10.3, 1.3], [1.0, 2.4, 3.5])), dim=-1)
#loss = calculate_kl_divergence(student_logits, teacher_probs)

#content_ranges = [[0, 3], [5, 8]]
#tokens = [43, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#content_tokens = [tokens[start:end] for start, end in content_ranges]
#print(content_tokens)
#print([tokens[start:end] for start, end in content_ranges])
#content_indices = [i+1 for start, end in content_ranges for i in range(start, end)]

#print(tokens[:content_indices[5]])

#print(torch.tensor([0.00000004, 4, 1]).numel)
#student_logits = torch.tensor([0.00000004, 4, 1])
#student_probs = torch.softmax(student_logits, dim=-1)
#student_probs_temp = student_probs ** (1 / temp)
#student_probs_temp = student_probs_temp / torch.sum(student_probs_temp)
#print(student_probs_temp)

#likes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#dislikes = [6, 3, 2]
#filtered = [i for i in likes if i not in dislikes]
#correct = [i - sum(j < i for j in dislikes) for i in likes if i not in dislikes]
#print(filtered)
#print(correct)
#print(torch.randperm(6).tolist())
#print(torch.tensor((0, 45)))
#[1, 3, 4, 2]


#list = [0, 1, 2, 3, 4, 5, 6]
#for i in list[:4]:
    #print(i)
#print([256 * 1536**2] + [0] * (2 - 1))
#tensor = torch.tensor(([0.0004, 10.1, 1.2], [1.5, 2.3, 3.4]))
#print(F.softmax(tensor, dim=-1))
#print(torch.exp(F.log_softmax(tensor, dim=0)))

#content_ranges = [[0, 3], [5, 8], [9, 10], [11, 14]]
#tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
#print([tokens[end - 1] for start, end in content_ranges]) # [2, 7, 9, 13]
#content_tokens = [token for start, end in content_ranges for token in tokens[start:end]]
#print(content_tokens) # [0, 1, 2, 5, 6, 7, 9, 11, 12, 13]
#max_len = 10
#content_ends = []

#current_id = -1
#for start, end in content_ranges:
    #if end > max_len:
        #break
    #content_ends.append(((end - start)) + current_id)
    #current_id = content_ends[-1]

#print(content_ends) # [2, 5, 6]
#print([content_tokens[end] for end in content_ends]) # [2, 7, 9] 

#tensor1 = torch.tensor(([0.00002, 0.00002, 0.99996], [0.00002, 0.00002, 0.99996]))
#tensor2 = torch.tensor(([2.0004, 5.3, 1.3], [1.0, 2.4, 3.5]))
#print(tensor1[0][0])
#print(torch.log(tensor1))
#print(calculate_kl_divergence(tensor1, tensor2, temp=1, custom=True))

#tensor3 = torch.tensor(([1,2,3], [1,2,3]), dtype=torch.float16)
#indices_to_select = torch.tensor(([1,2], [2,3]))
#tensor3 = torch.index
#print(tensor3)

#teacher = H5Reader(r"F:\distilled\soup\MythoMax-L2-Kimiko-v2-13b\validate", "cpu")
#teacher_tensor = teacher.read_next()
#print(calculate_divergence(teacher_tensor, teacher_tensor, temp=1, custom=True))
#teacher.close()

#kl_div = calculate_divergence(student_logits, teacher_tensor, temp=1, custom=False)
#print(kl_div)

#print(torch.tensor(([89])).exp())

# -103 tensor([1.4013e-45])
# 88 tensor([1.6516e+38])

#a = torch.tensor(([0.0004, 10.1, 1.2], [1.5, 2.3, 3.4]))
#print(torch.exp(F.log_softmax(a, dim=-1)))
#print(torch.exp(F.log_softmax(F.log_softmax(a, dim=-1)/2, dim=-1)))
#print(F.softmax(a/2, dim=-1))

#test tokenization
def input_prompt_format():
    prompt_format = {
        'SYS_START': "### System:\n",
        'USER_START': "### User:\n",
        'ASSISTANT_START': "### Assistant:\n",
        'SYS_END': '\n',
        'USER_END': '\n',
        'ASSISTANT_END': '\n'
    }
    print("Enter the prompt format")

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

#print(input_prompt_format())

#print( np.concatenate(([111,333,444,555], np.array([1] * 2, dtype=np.int64))))
# test cropping off pad tokens, if we only have the total number of pad tokens at the end of the conversation, pad token id = 0
#print(np.array([1, 2, 3, 4, 5, 6, 7, 0, 0, 0])[:-3])
#print(int(128 * 1000**2))

# cast np fp 32 array to fp 16
#arr = np.random.random(size=(10, 10))
#print(arr.dtype)
#print(arr[0][0].item())
#arrnp = arr.astype(np.float16)
#print(arrnp.dtype)
#print(arr[0][0].item())
## now same but with torch
#tensor = torch.tensor(arr)
#print(tensor.dtype)
#print(tensor[0][0].item())
#tensor = tensor.half()
#print(tensor.dtype)
#print(tensor[0][0].item())
#
#batch_list_np = [np.random.random(size=(10, 10)) for i in range(10)]
#batch_tensor = torch.tensor(batch_list_np)
#print(batch_tensor)

#for i in range(10):
    #print(i)
#
#import numpy as np
#import torch
#import time
#
## The monstrous dimensions we're still dealing with
#batch_size = 4
#tokens_per_convo = 1000
#features_per_token = 32000
#tokens_to_crop = 200  # Number of tokens we're gonna viciously rip out from the middle
#
## Middle point for the cropping, keeping it fair
#start = (tokens_per_convo - tokens_to_crop) // 2
#end = start + tokens_to_crop
#
## Creating the behemoths again
#numpy_array = np.random.rand(batch_size, tokens_per_convo, features_per_token).astype('float32')
#torch_tensor = torch.rand((batch_size, tokens_per_convo, features_per_token), dtype=torch.float32, device='cuda:0')
#
#def numpy_operations():
#    # Cropping like a boss
#    cropped = numpy_array[:, start:end, :]
#    # Let's do something nasty with the cropped section
#    result = np.log(cropped + 1) * 2.5
#    return np.max(result, axis=2)
#
#def torch_operations():
#    # Cropping with style
#    cropped = torch.index_select(torch_tensor, 1, torch.arange(start, end, device='cuda:0'))
#    # Equivalent nasty operations in PyTorch land
#    result = torch.log(cropped + 1) * 2.5
#    return torch.max(result, dim=2)
#
## Timing NumPy
#start_time = time.time()
#numpy_result = numpy_operations()
#numpy_time = time.time() - start_time
#
## Timing PyTorch
#start_time = time.time()
#torch_result = torch_operations()
#torch_time = time.time() - start_time
#
#print(f"NumPy took: {numpy_time} seconds")
#print(f"PyTorch took: {torch_time} seconds")
#
## Who's the slicing champ?
#if numpy_time < torch_time:
#    print("NumPy slices and dices faster.")
#else:
#    print("PyTorch cuts through the bullshit quicker.")

#logits = torch.tensor(([0.00004, 10.1, 1.2]), device='cuda:0', dtype=torch.float16)
#logits_temp = logits / 2
#log_probs = F.log_softmax(logits, dim=-1)
#
#print(log_probs)
#print(torch.log_softmax(logits_temp, dim=-1))
#mean = torch.mean(log_probs, dim=-1)
#print("###########")
#print(mean)
#print(mean/2)
#print((log_probs - mean))
#print((log_probs - mean)/2)
#print((log_probs/2))
#
#
#tensor1 = torch.tensor(([0.02, 0.02, 0.996], [0.2, 0.2, 0.996], [1, 2, 3])).numpy()
#tensor1_len = tensor1.shape[0]
#tensor2 = torch.tensor(([2.04, 5.3, 1.3], [1.0, 2.4, 3.5])).numpy()
#tensor2_len = tensor2.shape[0]

#full_distr = np.pad(tensor1, ((0, 1), (0, 0)))
#list1 = [1,2]
#list2 = [1,1]
#full_distr = tensor1[list1, list2]
#print(np.arange(20, 40))

#dict = {'a': 1, 'b': 2, 'c': 3}
#print(1 in dict)
#
#dict["d"] = 4
#print(dict.clear())
#list_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#for i in range(0, len(list_int), 3):
    #print(list_int[i:i+3])

#tensor = F.softmax(torch.tensor(([0.04, 10.1, 1.2], [1.5, 2.3, 3.4])), dim=-1)
# calculate entropy for each distribution
#print(tensor)
#for i in range(tensor.size(0)):
    #print(-torch.sum(tensor[i] * torch.log(tensor[i])))

#set1 = {1, 2, 3}
#set2 = {2, 3, 4}

#total_gpus = torch.cuda.device_count()
#for i in range(total_gpus):
    #gpu = {}
    #gpu["name"] = torch.cuda.get_device_name(i) + f" ({i})"
    #gpu["memory_total"] = torch.cuda.get_device_properties(i).total_memory/1024**2
    #gpu["memory_used"] = torch.cuda.memory_allocated(i)/1024**2
    #gpu["memory_free"] = torch.cuda.mem_get_info(i)[1]/1024**2
    #print(gpu["name"])
    #print(gpu["memory_total"])
    #print(gpu["memory_used"])
    #print(gpu["memory_free"])
#
#tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-7b", use_fast=False)
#
#all_tokens = tokenizer.get_vocab()
#print(tokenizer.convert_tokens_to_ids("<start_of_turn>"))
#added_tokens = ""
#print(added_tokens)
#base_tokens1 = set(all_tokens) - set(added_tokens)
#
#tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-gemma-v0.1")
#
#all_tokens = tokenizer.get_vocab().keys()
#added_tokens = tokenizer.all_special_tokens
#base_tokens2 = set(all_tokens) - set(added_tokens)
#
#print(base_tokens1 - base_tokens2)
#
#a = torch.tensor(([[1.5, 1], [2.3, 3.4]], [[1.5, 1], [2.3, 3.4]]))
#b = torch.tensor(([0.04, 10.1, 1.2], [1.5, 2.3, 3.4]))
# see how many elements are in the second dimension
#print(b.size(0))

def calculate_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, custom=True):
    # assert teacher_logits[0].sum() != 0, "Teacher logprobs are all zeros"

    reduction = 'none' if custom else 'batchmean'
    kl_div = F.kl_div(student_logits, teacher_logits, reduction=reduction, log_target=True)
    if custom:
        kl_div = ((kl_div.exp() - 1).sum(dim=-1) + 1).mean().log()

    return kl_div

def truncated_kldiv(student_probs, teacher_probs):
    sum = student_probs.sum(dim=-1).mean()

    div = F.mse_loss(student_probs, teacher_probs) * math.pow(3 - sum, 2)

    return div

#tensor1 = torch.tensor(([0.01, 0.99], [0.99, 0.01]), dtype=torch.float32).log()
#
#log_soft_1 = F.log_softmax(tensor1, dim=-1)
#
#indices = torch.tensor(([1, 0]))
#
#list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#print(list[1:-1])
#print(F.cross_entropy(log_soft_1, indices, reduction='none'))
#
#losses = torch.tensor(([0.01, 0.3]), dtype=torch.float32)
#power = 0.2
#add = 1
#weights = (losses/losses.max() + add).pow(power)
#transformed_weights = ((10/weights) + 1).pow(power)
#weighted_losses = losses * weights
#weighted_losses_transformed = losses * transformed_weights
#mean_losses = (weighted_losses + weighted_losses_transformed)/2
#print(f"Losses: [0.01, 0.3], power {power}, add {add}")
#print(f"Weights: ((losses/losses.max()) + {add}).pow({power}):", weights)
#print(f"Transformed weights: ((1/weights) + 1).pow(power): {transformed_weights}")
#print(f"Losses * Weights: {weighted_losses}")
#print(f"Losses * Transformed weights: {weighted_losses_transformed}")
#print(f"Mean of both: {mean_losses}")
#print(f"Total mean: {mean_losses.mean()}")
#
#dict1 = {1: 1, 'b': 2, 'c': 4, 'd': 5, 'a': 5}
#
#grouped_dict = defaultdict(list)
#for key, value in dict1.items():
#    grouped_dict[value].append(key)
#
#print(grouped_dict.items())
#print(dict1)
#
#model_path = r"C:\Users\PC\Desktop\TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
#
#model = LlamaForCausalLM.from_pretrained(
#    model_path,
#    device_map="auto",
#    attn_implementation="flash_attention_2"
#)
#
#layer_names = list(model.hf_device_map.keys())
#print(model.hf_device_map, "\n")
#del model
#
#torch.cuda.empty_cache()
#
#num_layers = len(layer_names)
#num_gpus = 3
#num_gpu0_layers = 2
#
## Distribute layers across GPUs
#custom_device_map = {}
#remaining_layers = num_layers - num_gpu0_layers
#layers_per_gpu = math.ceil(remaining_layers / (num_gpus - 1))
#
## Assign layers to GPU 0
#layer_index = 0
#for _ in range(num_gpu0_layers):
#    custom_device_map[layer_names[layer_index]] = 0
#    layer_index += 1
#
## Assign remaining layers to other GPUs
#gpu_id = 1
#while layer_index < num_layers:
#    for _ in range(layers_per_gpu):
#        if layer_index >= num_layers:
#            break
#        custom_device_map[layer_names[layer_index]] = gpu_id
#        layer_index += 1
#    gpu_id += 1
#
#print(custom_device_map)
#
#input("Press Enter to continue...")
#
#model = AutoModelForCausalLM.from_pretrained(
#    model_path,
#    device_map=custom_device_map,
#    attn_implementation="flash_attention_2"
#)
#print(model.hf_device_map)
#input("Press Enter to continue...")
#
#
#letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
#
#content_ranges = [[0, 3], [5, 8]]
#
#content_indices = []
#for start, end in content_ranges:
#    content_indices.extend(range(start, end))
#
#content_letters = [letter_list[i] for i in content_indices]
#
#inner_content_indices = []
#start_idx = 0
#
#for start, end in content_ranges:
#    end_idx = start_idx + (end - start) - 1
#    inner_content_indices.append([start_idx, end_idx])
#    start_idx = end_idx + 1
#
#print(inner_content_indices)
#
#
#logits = torch.tensor(([0.00004, 10.1, 1.2], [1.5, 2.3, 3.4]))
#
#logprobs = F.log_softmax(logits, dim=-1)
#
#print(logprobs.exp())
#
#temp = 2
#
#logprobs_temp = F.log_softmax(logprobs / temp, dim=-1)
#logits_temp = F.log_softmax(logits / temp, dim=-1)
#
#print(logprobs_temp.exp())
#print(logits_temp.exp())
#
#tokenizer = AutoTokenizer.from_pretrained("alpindale/gemma-7b", use_fast=False)
#
#
# prep and load an exllama model
#
#model_path = r"C:\Users\PC\Desktop\TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
#
#context_len = 2048
#batch_size = 1
#seq_chunk_len = 256
#
#config = ExLlamaV2Config()
#config.model_dir = model_path
#config.prepare()
#config.max_seq_len = context_len
#config.max_batch_size = batch_size
#config.max_input_len = seq_chunk_len
#config.max_attention_size = context_len ** 2 
#
#model = ExLlamaV2(config)
#tokenizer = ExLlamaV2Tokenizer(config)
#cache = ExLlamaV2Cache(model, batch_size, context_len, lazy=True)
#
#model.load_autosplit(cache, reserve_vram=[256 * 1536**2]*2)
#
#text = "Hello, how are you"
#
#tokenized = tokenizer.encode(text)
#
#forward_pass = model.forward(tokenized, cache=cache, last_id_only=True)
#
#sum = forward_pass.exp().sum()
#prob_sum = F.softmax(forward_pass, dim=-1).sum()
#print(sum)
#print(prob_sum)
#
## save the forward pass to a file
#with open("forward_pass.json", "w") as f:
#    json.dump(forward_pass.tolist(), f)
#
# open the forward pass files and make the variables for two of them
#
#with open("forward_pass1.json", "r") as f:
#    forward_pass1 = torch.tensor(json.load(f)).squeeze()
#
#with open("forward_pass2.json", "r") as f:
#    forward_pass2 = torch.tensor(json.load(f)).squeeze()
#
#eps = torch.tensor(1e-10)
#
#student_logprobs = F.log_softmax(forward_pass1, dim=-1)
#teacher_logits = F.log_softmax(forward_pass2, dim=-1)
#
#print(F.kl_div(student_logprobs, teacher_logits, reduction='batchmean', log_target=True))
#
#K = 200
#topK, indices = torch.topk(teacher_logits, K, dim=-1)
#
#rev_sum_topK = (1 - teacher_logits.exp().sum(dim=-1)).clamp(eps)
#num_pad_logits = student_logprobs.size(-1) - topK.size(-1)
#print(num_pad_logits)
#pad_logits = (rev_sum_topK / num_pad_logits).log()
#print(pad_logits)
#
#teacher_logits_padded = torch.zeros_like(student_logprobs) + pad_logits.unsqueeze(-1)
#
#teacher_logits_padded.scatter_(-1, indices, topK)
#
#teacher_logprobs = teacher_logits_padded
#print(teacher_logprobs)
#print(student_logprobs)
#KL_div = F.kl_div(student_logprobs, teacher_logprobs, reduction='batchmean', log_target=True)
#
#print(KL_div)
#
#
## Given data and indices tensors
#data = torch.tensor([[0.123, 24.3, 9.4], [0.62, 0.121, 53.23]])
#indices = torch.tensor([[6, 95, 2124], [934, 953, 11]])
#
## Sequence of IDs you want to retrieve
#sequence = torch.tensor([411, 11])
#
## Flatten the data and indices tensors
#flat_data = data.flatten()
#flat_indices = indices.flatten()
#
## Create a sparse tensor
#sparse_indices = flat_indices.unsqueeze(0)
#sparse_data = flat_data
#sparse_tensor = torch.sparse_coo_tensor(sparse_indices, sparse_data)
#
## Function to retrieve values from the sequence
#def retrieve_values(sequence, sparse_tensor):
#    dense_lookup = sparse_tensor.to_dense()
#    result = torch.zeros(sequence.size(), dtype=dense_lookup.dtype)
#    valid_indices = sequence < dense_lookup.size(0)
#    result[valid_indices] = dense_lookup[sequence[valid_indices]]
#    return result
#
## Retrieve the correct values
#retrieved_values = retrieve_values(sequence, sparse_tensor)
#print(retrieved_values)

