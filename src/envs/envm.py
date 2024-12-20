import torch
import platform
import sys

from src.envs.hello_world import hello_world

def print_system_environment():
   # 获取Python主版本号
    python_version = sys.version.split()[0]
    # 获取操作系统名称
    platform_name = platform.system()
    # 获取操作系统版本
    platform_version = platform.version()
    # 获取系统架构
    architecture = platform.architecture()[0]
    # 获取机器类型
    machine = platform.machine()
    # 环境信息，这里固定为"Development"，可按需修改
    environment = "Development"
    # 获取PyTorch版本号
    torch_version = torch.__version__

    # 依次打印各项配置信息
    print("Python Version:", python_version)
    print("Platform:", platform_name)
    print("Platform Version:", platform_version)
    print("Architecture:", architecture)
    print("Machine:", machine)
    print("Environment:", environment)
    print("Torch:", torch_version)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")

    hello_world()

