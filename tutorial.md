## HB-PyTorch Development Tutorial

### Setup HB-PyTorch Development Environment

Follow emulation layer setup guide in README
https://github.com/cornell-brg/hb-pytorch/tree/master#how-to-build-pytorch-with-emulation-layer

After successfully built Hb-PyTorch, you can open up a python console and give it try
NOTE: you cannot do `import torch` from top level of hb-pytorch dir, since there is a `torch` folder ...

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
>>> x + x
Emulation layer enqueued kernel tensorlib_add
  Emulation layer launched 16 threads to simulate the tile group
tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.], device='hammerblade')
```

Here `torch.ones(10)` means creating a 10-element tensor, and `.hammerblade()` moves the tensor from host memory to HB memory.

### Extend HB-PyTorch with a New Operator

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
