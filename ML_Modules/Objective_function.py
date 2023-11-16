import torch

# Define the function x1^2 + x2^2
def func_scalar(X):
    return torch.sum(X ** 2, dim = 1)

def func_vector(X):
    a = torch.sum(X ** 2, dim = 1)
    b = torch.sum(X ** 3, dim = 1)
    return torch.stack((a, b), dim = 1)