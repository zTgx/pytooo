import sys
import argparse
from envs import envm

def print_environment():
    print("Python version:", sys.version)
    print("Platform:", sys.platform)
    envm.print_cuda_env()
    
def main(print_env=True):
    if print_env:
        print_environment()

    # Logic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with optional environment printing.")
    parser.add_argument('--print-env', type=bool, default=True, help='Whether to print environment information.')
    
    args = parser.parse_args()
    
    main(print_env=args.print_env)
