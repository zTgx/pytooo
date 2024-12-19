import pandas as pd
import torch
import time

url = "https://raw.githubusercontent.com/kittenpub/database-repository/main/Fish_Dataset_Pytorch.csv"
fish_data = pd.read_csv(url)
print(fish_data.head())
fish_tensor = torch.tensor(fish_data.iloc[:, :-1].values, dtype=torch.float32)
print(f"Shape of the fish tensor: {fish_tensor.shape}")

reshaped_tensor = fish_tensor.view(-1, 2) # Reshape into 2 columns with
print(f"Reshaped Tensor (2 columns): {reshaped_tensor.shape}")

reshaped_tensor_batch = fish_tensor.view(10, -1)
print(f"Reshaped Tensor (10 rows): {reshaped_tensor_batch.shape}")

# Slicing the first 5 rows and the first 3 column
sliced_tensor = fish_tensor[:5, :3]

print(f"Sliced Tensor (First 5 rows, First 3 columns):\n{sliced_tensor}")

mean_tensor = torch.mean(fish_tensor, dim=0)
sum_tensor = torch.sum(fish_tensor, dim=1)
print(f"Mean Tensor (Column-wise): {mean_tensor}")
print(f"Sum Tensor (Row-wise): {sum_tensor}")

scalar = torch.tensor(10.0)

broadcasted_tensor = fish_tensor + scalar

mean = fish_tensor.mean(dim=0, keepdim=True)
std = fish_tensor.std(dim=0, keepdim=True)

normalized_tensor = (fish_tensor - mean) / std

print(f"Normalized Tensor:\n{normalized_tensor}")

# Move the tensor to the GPU using CUDA
fish_tensor_gpu = fish_tensor.cuda()

print(f"Is the tensor on GPU? {fish_tensor_gpu.is_cuda}")

reshaped_tensor_gpu = fish_tensor_gpu.view(-1, 2)

scalar_gpu = torch.tensor(10.0).cuda()

start_gpu = time.time()
end_gpu = time.time()

# 高级操作
# Stacking Tensors
# Squeezing and Unsqueezing Tensors
# Permuting Tensors
# Permuting Dimensions

# CNNs (Convolutional Neural Networks)
# Feedforward Neural Networks (FNNs)
# multi-layer perceptron (MLP),
# Recurrent Neural Networks (RNNs)
# Stochastic Gradient Descent (SGD)
# Adaptive Moment Estimation (Adam)
# Automatic Mixed Precision (AMP)


# Unlike RNNs, which process sequential data step by step and are limited
# by their sequential nature, transformers handle entire sequences in
# parallel.

# BERT (Bidirectional Encoder Representation
# NLP
# Multi-Head Transformer
# Positional Since transformer


# torch.compile() is a new feature introduced in PyTorch 2.0 aimed at
# improving the performance of neural networks by optimizing the
# computational graph.

# PyTorch Quantization

# Model quantization is a crucial technique in deep learning that enables
# efficient deployment of neural networks

# PTQ and QAT

# Mixed Precision Training



# ONNX
# directed acyclic graph (DAG)
# MobileNetV2,
# TorchServe simplifies this process, enabling
# developers to serve PyTorch models as RESTful APIs for real-time
# inference.

