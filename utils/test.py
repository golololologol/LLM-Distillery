from sympy import content
import torch
import torch.nn as nn
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import math
import os
import threading
import queue
import h5py
from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer



def calculate_kl_divergence(student_logits, teacher_probs, temp=1, per_token=True):
    student_log_probs = nn.functional.log_softmax(student_logits/temp, dim=-1)
    reduction = 'batchmean' if not per_token else 'sum'
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return kl_div

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
content_ranges = [[0, 3], [5, 8], [9, 10], [11, 14]]
tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
print([tokens[end - 1] for start, end in content_ranges]) # [2, 7, 9, 13]
content_tokens = [token for start, end in content_ranges for token in tokens[start:end]]
print(content_tokens) # [0, 1, 2, 5, 6, 7, 9, 11, 12, 13]
max_len = 10
content_ends = []

current_id = -1
for start, end in content_ranges:
    if end > max_len:
        break
    content_ends.append(((end - start)) + current_id)
    current_id = content_ends[-1]

print(content_ends) # [2, 5, 6]
print([content_tokens[end] for end in content_ends]) # [2, 7, 9] 