import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import torch.nn.functional as F
import json
import codecs
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

#print(base_tokens1 - base_tokens2)

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

#log_soft_1 = F.log_softmax(tensor1, dim=-1)

#indices = torch.tensor(([1, 0]))

#list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#print(list[1:-1])
#print(F.cross_entropy(log_soft_1, indices, reduction='none'))

print(0.25 ** 0.5)