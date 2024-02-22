import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
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

def calculate_divergence(student_logits, teacher_log_probs: torch.Tensor, temp=1.0, custom=False):
    student_log_probs = F.log_softmax(F.log_softmax(student_logits, dim=-1).clip(-103, 88) / temp, dim=-1)
    
    reduction = 'batchmean' if not custom else 'none'
    kl_div = F.kl_div(student_log_probs, F.log_softmax(F.log_softmax(teacher_log_probs, dim=-1).clip(-103, 88) / temp, dim=-1), reduction=reduction, log_target=True)
    if custom:
        kl_div = ((kl_div.exp() - 1).sum(dim=-1) + 1).mean().log()

    return kl_div*temp


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