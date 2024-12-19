import torch

def hello_world():
    '''
        最小 torch + cuda 功能验证
    '''
    x = torch.rand(5, 3)
    x = x.cuda() # 移到 GPU 上
    y = torch.rand(3, 3).cuda()
    result = torch.matmul(x, y)
    # print(result)

    print("Tensor is on GPU:", result.is_cuda)