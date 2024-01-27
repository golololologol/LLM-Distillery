import torch
import torch.nn as nn
import numpy as np
import h5py
import queue

def calculate_kl_divergence(student_logits, teacher_probs, temp=1):
    student_log_probs = nn.functional.log_softmax(student_logits/temp, dim=-1)
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_div

d1 = np.random.random(size = (10,2))
d2 = torch.tensor(([1]), dtype=torch.float16)
q = queue.Queue()
q.put(d1)
q.put(d2)
with h5py.File('test.h5', 'a') as hf:
    count = 0
    while not q.empty():
        data = q.get()
        # make a new dataset if it doesn't exist yet
        if f'data_{count}' not in hf:
            hf.create_dataset(f'data_{count}', data=data)
        else:
            # otherwise, rewrite the existing dataset
            hf.__delitem__(f'data_{count}')
            hf.create_dataset(f'data_{count}', data=data)
        count += 1

with h5py.File('test.h5', 'r') as hf:
    print(hf.keys())
    print(torch.tensor(hf['data_0']))
    print(torch.tensor(hf['data_1']))
#temp = 2
#student_logits = torch.tensor(([0.0004, 10.3, 1.3], [1.0, 2.4, 3.5]))
#teacher_probs = nn.functional.softmax(torch.tensor(([0.0004, 10.3, 1.3], [1.0, 2.4, 3.5])), dim=-1)

#print(calculate_kl_divergence(student_logits, teacher_probs, temp=temp))
#print(torch.tensor([0.00000004, 4, 1]).numel)
#student_logits = torch.tensor([0.00000004, 4, 1])
#student_logits_temp = torch.softmax(student_logits / temp, dim=-1)
#print(student_logits_temp)

#student_probs = torch.softmax(student_logits, dim=-1)
#student_probs_temp = student_probs ** (1 / temp)
#student_probs_temp = student_probs_temp / torch.sum(student_probs_temp)
#print(student_probs_temp)