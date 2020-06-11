![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

## HB-PyTorch Development Tutorial
 - Author: Lin Cheng (lc873@cornell.edu)
 - Date: June 11, 2020
 
#### Table of Contents
 - Introduction
 - Setup HB-PyTorch Development Environment with Emulation Layer
 - Vector-Increment Operator
 - (Fake) Auto Differentiation for Vector-Increment
 - To Do On Your Own (Vector-Vector-Add)
 
### Introduction
This tutorial will discuss how to setup development environemnt for HB-PyTorch, add a branch new operator to PyTorch for HammerBlade, and register auto-differentiation (backward pass) for your new operator. We start with HammerBlade _Emulation_Layer_, in which all interaction with HammerBlade hardware is emualted natively on x86. Similarly, HammerBlade device kernels are compiled for, and executed natively on x86 as well. If you have experience with iOS or Android development, building for Emulation Layer is analog to building for simulator in XCode.

Here is a dataflow graph from PyTorch wiki. This graph is arguably out dated and it looks quite scary. But it looks cool ;)
[![PyTorch Data Flow and Interface Diagram](https://raw.githubusercontent.com/wiki/pytorch/pytorch/images/pytorch_wiki_dataflow_interface_diagram.png)](https://github.com/pytorch/pytorch/wiki/PyTorch-Data-Flow-and-Interface-Diagram)

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
Now we try extending PyTorch by adding a new operator -- vincr. `vincr` takes a single tensor, and adds one to each element.
```
python
Python 3.7.4 (default, Sep 28 2019, 14:22:12)
[GCC 6.3.1 20170216 (Red Hat 6.3.1-3)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> x = torch.ones(10).hammerblade()
Emulating CUDALite...
Emulation barrier init'ed with 4 threads
PyTorch configed with 2 * 2 HB device
HB startup config kernel applied
>>> torch.vincr(x)
tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.], device='hammerblade')
```
The first step of adding a new operator is registering it with ATen.

Edit [aten/src/ATen/native/native_functions.yaml](aten/src/ATen/native/native_functions.yaml) with your favorite text editor (vim) and add the following to the end of the file
```yaml
- func: vincr(Tensor self) -> Tensor
  use_c10_dispatcher: full
  variants: function
  dispatch:
    HammerBlade: vincr_hb
```
Here we declare a new ATen operator `vincr` which takes in a single tenosr `self` as argument (`vincr(Tensor self)`). `-> Tensor` indicates our `vincr` should return a tensor. The next thing to notice is the very last two lines.
```yaml
  dispatch:
    HammerBlade: vincr_hb
```
These 2 lines tell the ATen dispatcher that, if `vincr` is called and the input tensor is a HammerBlade tensor, a function named `vincr_hb` should be called. More info on advanced registration can be found in [aten/src/ATen/native/README.md](aten/src/ATen/native/README.md).

Next we create the _host function_ (i.e., `vincr_hb`) which is in charge of performing the offloading.

Edit [aten/src/ATen/native/hammerblade/Vincr.cpp](aten/src/ATen/native/hammerblade/Vincr.cpp) and add the following lines to it.
```cpp
#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor vincr_hb(const Tensor& self) {

  // Create an output tensor that has the same shape as self
  auto result = at::empty_like(self, self.options());

  // Call HB device kernel tensorlib_vincr
  hb_offload_kernel(result, self, "tensorlib_vincr");

  return result;
}

}} // namespace at::native
```
Just like `vincr` we declared in [aten/src/ATen/native/native_functions.yaml](aten/src/ATen/native/native_functions.yaml), `vincr_hb` also takes a tensor as argument and returns a tensor. The main functionality of `vincr_hb` is to create the return tensor and invoke the actual compute kernel on HB by calling `hb_offload_kernel`.

To read more about offloading helper functions, you may refer to [aten/src/ATen/native/hammerblade/Offload.h](aten/src/ATen/native/hammerblade/Offload.h).

Finally, we create the compute kernel that runs on HammerBlade, and performs the +1 operation.

Edit [hammerblade/torch/kernel/kernel_vincr.cpp](hammerblade/torch/kernel/kernel_vincr.cpp) and add the following lines.
```cpp
#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vincr(
          hb_tensor_t* result_p,
          hb_tensor_t* self_p) {

    // Convert low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {
      // Add 1 to each element
      for (size_t i = 0; i < self.numel(); i++) {
        result(i) = self(i) + 1;
      }
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vincr, hb_tensor_t*, hb_tensor_t*)

}
```
This piece of code appears to be complicated ... but it's very structured.
```cpp
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);
 ```
Here we wrap low-level tensor struct objects to our `HBTensor` helper class objects. You may refer to [hammerblade/torch/kernel/hb_tensor.hpp](hammerblade/torch/kernel/hb_tensor.hpp) for its internals.
```cpp
bsg_cuda_print_stat_kernel_start();
...
bsg_cuda_print_stat_kernel_end();
```
These 2 functions mark the start and end of a kernel. They are meant for performance profiling when running on COSIM. With emulation layer, they are simply NOPs.
```cpp
    if (__bsg_id == 0) {
      // Add 1 to each element
      for (size_t i = 0; i < self.numel(); i++) {
        result(i) = self(i) + 1;
      }
    }
```
This is the core of `vincr`. `if (__bsg_id == 0)` makes sure only tile 0 does the work.
```cpp
g_barrier.sync();
```
A barrier at the end of every kernel is necessary, since HammerBlade does not wait for all tiles to finish. (which is a little wired.)
```cpp
HB_EMUL_REG_KERNEL(tensorlib_vincr, hb_tensor_t*, hb_tensor_t*)
```
Finally, we need to register this kernel with emulation layer, so we can lookup it by its name when we do the offloading. Recall that in the host code we used a string to locate the HammerBlade kernel function
```cpp
   hb_offload_kernel(result, self, "tensorlib_vincr");
```
Now we are ready to re-build PyTorch and give it a try. You must re-build PyTorch with `--cmake` flag since we added new files.
```bash
python setup.py develop --cmake
```
Finally, we add tests on our newly created `vincr` operator.

Edit [hammerblade/torch/tests/test_vincr.py](hammerblade/torch/tests/test_vincr.py) and add the following lines.
```python
import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_vincr(tensor):
    h = tensor.hammerblade()
    out = torch.vincr(h)
    assert out.device == torch.device("hammerblade")
    assert torch.allclose(tensor + 1, out.cpu())

def test_torch_vincr_1():
    t = torch.ones(10)
    _test_torch_vincr(t)

def test_torch_vincr_2():
    t = torch.randn(10)
    _test_torch_vincr(t)

def test_torch_vincr_3():
    t = torch.randn(2, 3)
    _test_torch_vincr(t)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_vincr_hypothesis(tensor):
    t = torch.tensor(tensor)
    _test_torch_vincr(t)
```

Our tests, very much like `test_vincr.py`, are composed by a few direct tests (`test_torch_vincr_1~3`) and a hypothesis test (`test_torch_vincr_hypothesis`). [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) is a widely adopted property-based testing framework. We have a few wrappers around hypothesis to make writing tests simpler. You may refer to [hammerblade/torch/tests/hypothesis_test_util.py](hammerblade/torch/tests/hypothesis_test_util.py) for more info.

Direct tests are relatively simple to parse, so we focus on the hypothesis test.
```python
@settings(deadline=None)
```
This line is necessary since running on COSIM take a long time and hypothesis by default has a very low timeout threshold.
```python
@given(tensor=hu.tensor())
```
This line tells hypothesis to generate a single "tensor". It's actually a numpy array ... that's why we need to manually convert it to a PyTorch tensor
```python
t = torch.tensor(tensor)
```


To run all tests, go to [hammerblade/torch](hammerblade/torch) and run
```bash
python pytest_runner.py
```

### (Fake) Auto Differentiation for Vector-Increment
One nice thing about PyTorch is that when writing a model, the user does not need to specify the backward pass. _Auto differentiation_ takes care of it. However, when a brand new operator is added to PyTorch, there is (at least currently) no way to magically figure out what is differential of your new operator -- it's not so _auto_ after all ...

As developers, we need to tell PyTorch how to handle our `vincr` when "automatically" generates the backward pass path.

To do so, edit [tools/autograd/derivatives.yaml](tools/autograd/derivatives.yaml) and add these to the end of the file
```yaml
- name: vincr(Tensor self) -> Tensor
  self: zeros_like(self).fill_(42)
```
It's easy to spot that the `name` field is the same as what we put down in [aten/src/ATen/native/native_functions.yaml](aten/src/ATen/native/native_functions.yaml) for `func`.
```yaml
  self: zeros_like(self).fill_(42)
```
This is the most important line. It says the gradient of `self` should be a tensor that has the same shape as self, and each element should be 42.

Now we re-build and give it a try. This time `--cmake` is not necessary since we only touched existing files.
```bash
python setup.py develop
```

```
python
Python 3.7.4 (default, Sep 28 2019, 14:22:12)
[GCC 6.3.1 20170216 (Red Hat 6.3.1-3)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> x = torch.ones(2, 2, requires_grad=True)
>>> h = x.hammerblade()
>>> y = torch.vincr(h)
>>> z = y.sum()
>>> z
tensor(8., device='hammerblade', grad_fn=<SumBackward0>)
>>> z.backward()
>>> x.grad
tensor([[42., 42.],
        [42., 42.]])
```
