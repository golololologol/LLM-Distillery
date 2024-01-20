import torch
import torch.nn as nn

distribution1 = torch.tensor([[2, -3, 0, 0.2]])
distribution2 = torch.tensor([[2.2, -2.5, 0.2, 0.1]])

distribution_probs1= nn.functional.softmax(distribution1, dim=1)
print(distribution_probs1)

distribution_probs2= nn.functional.softmax(distribution2, dim=1)
print(distribution_probs2)

distribution_probs_log1 = torch.log(distribution_probs1)
distribution_probs_log2 = torch.log(distribution_probs2)

distribution_merged= torch.stack([distribution_probs_log1, distribution_probs_log2]).mean(dim=0)

distribution_merged_exponent = torch.exp(distribution_merged)

distribution_merged_exponent_normalized = distribution_merged_exponent / distribution_merged_exponent.sum()
print(distribution_merged_exponent_normalized)

# result:
# tensor([[0.7649, 0.0052, 0.1035, 0.1264]])
# tensor([[0.7893, 0.0072, 0.1068, 0.0967]])
# tensor([[0.7779, 0.0061, 0.1053, 0.1107]])

