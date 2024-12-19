import torch


def print_cuda_env():
    x = torch.rand(5, 3)
    x = x.cuda()
    y = torch.rand(3, 3).cuda()
    result = torch.matmul(x, y)
    # print(result)

    print("Tensor is on GPU:", result.is_cuda)
    print(torch.__version__)

'''
2.5.1+cu124
'''