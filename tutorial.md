![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

## HB-PyTorch Development Tutorial
 - Author: Lin Cheng (lc873@cornell.edu)
 - Date: June 11, 2020
 
#### Table of Contents
 - Introduction
 - Setup HB-PyTorch Development Environment with Emulation Layer
 - Vector-Increment Operator
 - Vector-Vector-Add Operator
 - (Fake) Auto Differentiation for VVAdd 
 - To Do On Your Own
 
### Introduction
This tutorial will discuss how to setup development environemnt for HB-PyTorch, add a branch new operator to PyTorch for HammerBlade, and register auto-differentiation (backward pass) for your new operator. We start with HammerBlade _Emulation_Layer_, in which all interaction with HammerBlade hardware is emualted natively on x86. Similarly, HammerBlade device kernels are compiled for, and executed natively on x86 as well. If you have experience with iOS or Android development, building for Emulation Layer is analog to building for simulator in XCode.

### Setup HB-PyTorch Development Environment
Follow these steps to setup a development environment. Note: you need GCC > 7 to build PyTorch.
- Create a [Python virtual environment][venv]:

      python3 -m venv ./venv_pytorch
      source ./venv_pytorch/bin/activate

- Install some dependencies:

      pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing sklearn tqdm pytest ninja hypothesis

- Clone this repository:

      git clone git@github.com:cornell-brg/hb-pytorch.git
      
- Change directory:

      cd hb-pytorch

- Init PyTorch third party dependencies:

      git submodule update --init --recursive

- Setup building environment variables:

      source setup_emul_build_env.sh

- Build PyTorch. This step can take up to 15 minutes:

      python setup.py develop

- Turn on emulation debug info

      export HBEMUL_DEBUG=1

- Setup emulated HB device size

      export HBEMUL_TILE_X_DIM=2
      export HBEMUL_TILE_Y_DIM=2
      
- Varify PyTorch with HB support is correctly built. Note: you cannot perform `import torch` at the top level of this repo, since there is a directory named `torch`

      cd ../
      python
      Python 3.7.4 (default, Sep 28 2019, 14:22:12)
      [GCC 6.3.1 20170216 (Red Hat 6.3.1-3)] on linux
      >>> import torch
      >>> torch.hammerblade.init()
      Emulating CUDALite...
      Emulation barrier init'ed with 4 threads
      PyTorch configed with 2 * 2 HB device
      Emulation layer enqueued kernel tensorlib_hb_startup
      Emulation layer launched 1 threads to simulate the tile group
      HB startup config kernel applied
      >>> x = torch.ones(10).hammerblade()
      >>> x
      tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='hammerblade')
      >>> x + x
      Emulation layer enqueued kernel tensorlib_add
      Emulation layer launched 1 threads to simulate the tile group
      tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.], device='hammerblade')
      
[venv]: https://docs.python.org/3/tutorial/venv.html

### Vector-Increment Operator
Now we try extending PyTorch by adding a new operator, which operates on HammerBlade 

In this section we add a new operator -- vincr -- to pytorch.
To do so, we need to modify 3 files
`aten/src/ATen/native/native_functions.yaml` -- to register the operation
`aten/src/ATen/native/hammerblade/Vincr.cpp` -- vincr kernel host code (runs on CPU)
`hammerblade/torch/kernel/kernel_vincr.cpp` -- vincr kernel device code (runs on HB)

Take a look at these 3 files and search for `vincr` if the file is too long.

```python
Python 3.6.8 (default, May  2 2019, 20:40:44)
[GCC 4.8.5 20150623 (Red Hat 4.8.5-36)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> x = torch.ones(10).hammerblade()
Emulating CUDALite...
Emulation barrier init'ed with 16 threads
PyTorch configed with 4 * 4 HB device
Emulation layer enqueued kernel tensorlib_hb_startup
  Emulation layer launched 16 threads to simulate the tile group
HB startup config kernel applied
>>> torch.vincr(x)
Emulation layer enqueued kernel tensorlib_vincr
  Emulation layer launched 16 threads to simulate the tile group
tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.], device='hammerblade')
```

### Tutorial Task -- Extend PyTorch with a Vector-Vector Add Operator

Similarly, we need to edit 3 files
`aten/src/ATen/native/native_functions.yaml` -- to register the operation
`aten/src/ATen/native/hammerblade/Vvadd.cpp` -- vincr kernel host code (runs on CPU)
`hammerblade/torch/kernel/kernel_vadd.cpp` -- vincr kernel device code (runs on HB)

Templates are added for you. Search for `Tutorial TODO` to find the tasks.

Good luck!
