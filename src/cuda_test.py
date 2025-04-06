#!/usr/bin/env python3

import torch
import subprocess

def check_torch_cuda():
    """Check CUDA availability using PyTorch."""
    if torch.cuda.is_available():
        print("CUDA is available in PyTorch.")
        num_devices = torch.cuda.device_count()
        print(f"Number of GPUs detected by PyTorch: {num_devices}")
        for i in range(num_devices):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available in PyTorch.")

def check_nvidia_smi():
    """Check GPU details using the nvidia-smi command."""
    try:
        output = subprocess.check_output(['nvidia-smi'], univeUsing device: cpu
Output shape: torch.Size([1, 16, 32, 32])
rsal_newlines=True)
        print("\nOutput from nvidia-smi:\n")
        print(output)
    except FileNotFoundError:
        print("nvidia-smi command not found. Is the NVIDIA driver installed?")
    except Exception as e:
        print(f"An error occurred while running nvidia-smi: {e}")

def main():
    print("=== CUDA Check ===\n")
    print("Checking CUDA availability using PyTorch:")
    check_torch_cuda()
    
    print("\nChecking GPU details with nvidia-smi:")
    check_nvidia_smi()

if __name__ == "__main__":
    main()
