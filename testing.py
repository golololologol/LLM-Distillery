import torch

num_devices = torch.torch.cuda.device_count()
print(f"Number of CUDA devices: {num_devices}")