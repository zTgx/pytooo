import torch
import platform
import sys
import pprint

from envs.hello_world import hello_world

def print_system_environment():
    # 动态生成配置信息
    config = {
        "Python Version": sys.version.split()[0],  # 获取 Python 主版本号
        "Platform": platform.system(),               # 操作系统名称
        "Platform Version": platform.version(),      # 操作系统版本
        "Architecture": platform.architecture()[0],  # 系统架构
        "Machine": platform.machine(),                # 机器类型
        "Environment": "Development",                 # 环境信息，可以根据需要修改
        "Torch": torch.__version__,                  # 2.5.1+cu124
    }

    # 使用 PrettyPrinter 美化输出
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    hello_world()
