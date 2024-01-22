import torch
import torch.nn as nn

def calculate_kl_divergence(student_probs, teacher_probs):
    epsilon = 1e-6
    student_log_probs = torch.log(student_probs) + epsilon
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_div

student_probs = torch.tensor([0.000005, 0.000005, 0.99999])
teacher_probs = torch.tensor([0.99999, 0.000005, 0.000005])

print(calculate_kl_divergence(student_probs, teacher_probs))