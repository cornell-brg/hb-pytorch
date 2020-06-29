import torch
x = torch.randn(10678,16,4)
h = x.hammerblade()

torch.hammerblade.profiler.route.add("at::Tensor at::CPUType::{anonymous}::add(const at::Tensor&, const at::Tensor&, c10::Scalar)", True)
#torch.hammerblade.profiler.route.add("at::Tensor at::CPUType::{anonymous}::mul(const at::Tensor&, const at::Tensor&)", True)

torch.hammerblade.profiler.enable()
y = x + x
torch.hammerblade.profiler.disable()
