import time
import numpy as np
import torch
import add2
# from torch.utils.cpp_extension import load

# cuda_module = load(name="add2",
#                    sources=["kernel/add2.cpp", "kernel/add2_kernel.cu"],
#                    verbose=True)

# c = a + b (shape: [n])
n = 4
# a = torch.rand(n, device="cuda:0")
# b = torch.rand(n, device="cuda:0")
a = torch.tensor([2.0,7.0,1.0,8.0], device="cuda:0")
b = torch.tensor([3.0,1.0,4.0,1.0], device="cuda:0")
cuda_c = torch.rand(n, device="cuda:0")
print(cuda_c)
ntest = 10

def show_time(func):
    times = list()
    res = list()
    # GPU warm up
    for _ in range(10):
        func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        r = func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()

        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times, res

def run_cuda():
    add2.torch_launch_add2(cuda_c, a, b, n)
    return cuda_c

def run_torch():
    # return None to avoid intermediate GPU memory application
    # for accurate time statistics
    a + b
    return None

print("Running cuda...")
cuda_time, _ = show_time(run_cuda)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running torch...")
torch_time, _ = show_time(run_torch)
print("Torch time:  {:.3f}us".format(np.mean(torch_time)))


print(cuda_c)
print((a+b))


