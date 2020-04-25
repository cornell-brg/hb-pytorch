import torch

torch.aten_profiler_start()
x = torch.randn(10)
print(x)
torch.aten_profiler_end()
