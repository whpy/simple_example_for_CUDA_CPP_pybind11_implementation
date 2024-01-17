This repository is mainly refers https://github.com/godweiyang/NN-CUDA-Example. We extract a very simple example from it to demonstrate how to write an interface between CUDA/C++ and python through the tools provided by pytorch. 

# Environment
## System
|dependency | version|
|---|---|
|Ubuntu| 22.04.1 LTS |
|Nvidia Driver| 537.13 |
|nvcc| V12.3.103 |
|gcc| 11.3.0 |

## Python
|dependency | version|
|---|---|
|Python| 3.10.10 |
|setuptools| 65.6.3 |
|torch| 2.1.1 |
|ninja | 1.11.1.1 |
|pybind11 | 2.11.1|


# Usage
Suppose that the original file structure is as below:
```.
├── README.md
├── include
│   └── add2.h
├── kernel
│   ├── add2.cpp
│   └── add2_kernel.cu
├── setup.py
└── test.py
```

## 1. Compile
run the following command(no need for su at least under conda environments)
```bash
python3 setup.py install
```
in `./`. We might see the output something like
```
Compiling objects...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/2] /usr/local/cuda-12.3/bin/nvcc  ...
[2/2] c++ ...
creating build/lib.linux-x86_64-cpython-310
g++ -pthread -B ...
```
we hide some details to protect privacy. The files structure would then become:
```
.
├── README.md
├── add2.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── build
│   ├── bdist.linux-x86_64
│   ├── lib.linux-x86_64-cpython-310
│   │   └── add2.cpython-310-x86_64-linux-gnu.so
│   └── temp.linux-x86_64-cpython-310
│       ├── build.ninja
│       └── kernel
│           ├── add2.o
│           └── add2_kernel.o
├── dist
│   └── add2-0.0.0-py3.10-linux-x86_64.egg
├── include
│   └── add2.h
├── kernel
│   ├── add2.cpp
│   └── add2_kernel.cu
├── setup.py
└── test.py
```

## 2. Test
after compilation, we run:
```python
python3 test.py
```
we would get:
```
tensor([0.1686, 0.2939, 0.1458, 0.3179], device='cuda:0')
Running cuda...
Cuda time:  93.484us
Running torch...
Torch time:  168.467us
tensor([5., 8., 5., 9.], device='cuda:0')
tensor([5., 8., 5., 9.], device='cuda:0')
```
we can modify the size of a,b,c to further compare the efficiency of the function.

## 3. Recompile
After adding some features or modification, we should type following commands to clear the temporary files and rebuild the modules. Then reinstall by pip:
```shell
python3 setup.py clean --all
python3 setup.py build
pip install .
```
