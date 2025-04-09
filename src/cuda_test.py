import torch

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPU(s):")
        
        # List each detected GPU device and its name
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        
        # Choose the first GPU device
        device = torch.device("cuda:0")
        
        # Create sample tensors on the GPU and perform a simple addition
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([10.0, 20.0, 30.0], device=device)
        result = a + b
        
        # Print the result of the tensor addition
        print("Tensor addition result on GPU:", result)
    else:
        print("CUDA is not available. Please check your Docker GPU settings.")

if __name__ == '__main__':
    main()