import torch

# 一个标量（scalar）的秩为 0，因为它没有维度；
# 一个向量（vector）的秩为 1，它有一个维度；
# 一个矩阵（matrix）的秩为 2，有两个维度（行和列）；
# 对于更高阶的张量，秩相应地表示其维度的个数
def rank():
    a = torch.tensor([1, 2, 3])  # 创建一个向量
    print(a.ndim)  # 输出1，因为向量的秩为1

# 形状描述了张量在每个维度上的长度
def shape():
    b = torch.tensor([[1, 2], [3, 4]])  # 创建一个2x2的矩阵
    print(b.shape)  # 输出 (2, 2)，表示这个矩阵有2行2列

# 大小通常是指张量中元素的总数。它可以通过将形状中的所有维度长度相乘来计算
def size():
    c = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 创建一个2x3的矩阵
    print(c.numel())  # 输出6，因为这个矩阵共有2 * 3 = 6个元素

def broadcasting():
    # 创建一个形状为(3, 1)的张量
    a = torch.tensor([[1], [2], [3]])
    # 创建一个形状为(1, 4)的张量
    b = torch.tensor([[4, 5, 6, 7]])

    # 进行加法运算，会触发广播机制
    c = a + b

    print(c)

def empty():
    empty_tensor = torch.empty(3, 2)
    print(empty_tensor)

def zero():
    zero_tensor = torch.zeros(3, 2)
    print(zero_tensor)


def one():
    ones_tensor = torch.ones(3, 2)
    print(ones_tensor)

def random():
    # on the interval [0, 1)
    random_tensor = torch.rand(3, 2)
    print(random_tensor)

def data_type():
    # The default data type for tensors is 32-bit floating point.
    tensor = torch.ones(3, 2)
    print(tensor)
    print("Data Type: ", tensor.dtype)
    tensor = tensor.to(torch.float64)
    print("Data Type: ", tensor.dtype)


def with_type():
    tensor1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    print("Tensor 1:", tensor1)

def operations():
    tensor1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    tensor2 = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

    # Add
    result = tensor1 + tensor2
    print("Addition Result: ", result)

    # Subtraction
    result = tensor1 - tensor2
    print("Subtraction Result: ", result)

    # Multiplication (Element-wise)
    result = tensor1 * tensor2
    print("Multiplication Result: ", result)

    # Division
    result = tensor1 / tensor2
    print("Division Result: ", result)

def reshaping():
    tensor = torch.arange(9)
    print("Original Tensor:")
    print(tensor)
    # Reshape the tensor
    reshaped_tensor = tensor.view(3, 3)
    print("\nReshaped Tensor:")
    print(reshaped_tensor)

    # slicing
    sliced_tensor = reshaped_tensor[0:2, 0:2]
    