import torch
import pickle

def load_distributions(file_path):
    with open(file_path, 'rb') as f:
        distributions_numpy = pickle.load(f)
    distributions_tensors = [torch.from_numpy(dist) for dist in distributions_numpy]
    return distributions_tensors

def check_loaded_distributions(distributions):
    print(distributions[0][5])
    print(distributions[0].shape)

# Usage Example
distributions = load_distributions(r"F:\distilled\randoBS_neural-chat-7b-v3-1-exl2\distributions_1.pkl")
check_loaded_distributions(distributions)