import torch
import torch.nn.functional as F
import json
import pytest

def convert_dense_input(x, r, s):
    stride = (1, 1)
    kernel_size = (r, s)
    assert x.dim() == 4
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = torch.flatten(x, start_dim = 4)
    x = x.transpose(1, 3).transpose(1, 2)
    x = torch.flatten(x, start_dim = 3)
    x = torch.flatten(x, start_dim = 0, end_dim = 2).t().contiguous()
    return x

def load_conv2_sparse_weight():
    model = torch.load("LeNet_5.prune.conv.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv2_weight = weights.get('conv2.weight').cpu()
    return conv2_weight

def test_lenet5_sparse01_conv2():
    torch.hammerblade.init()
    di = torch.rand(1, 20, 12, 12)
    cpu_sw = load_conv2_sparse_weight().to_sparse()
   # cpu_i = convert_dense_input(di, 5, 5)
   # cpu_sw = load_conv2_sparse_weight().view(50, -1).to_sparse()
   # cpu_out = torch.sparse.mm(cpu_sw, cpu_i)
   # out1 = cpu_out.view(1, 50, 8, 8)
   # torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::to(const at::Tensor&, const c10::TensorOptions&, bool, bool, c10::optional<c10::MemoryFormat>)")
   # torch.hammerblade.profiler.chart.add("at::Tensor at::TypeDefault::conv2d(const at::Tensor&, const at::Tensor&, const at::Tensor&, c10::IntArrayRef, c10::IntArrayRef, c10::IntArrayRef, int64_t)")
    with open('sparse_conv2.json') as route:
        data = json.load(route)
    data = json.loads(route)
    torch.hammerblade.profiler.route.set_route_from_json(data)
    torch.hammerblade.profiler.enable()
#    hb_i = di.hammerblade()
#    hb_sw = cpu_sw.hammerblade()
    hb_out = F.conv2d(di, cpu_sw, bias = None, stride = 1, padding = 0, dilation = 1)
    torch.hammerblade.profiler.disable()
    #print(torch.hammerblade.profiler.exec_time.raw_stack())
    #print(torch.hammerblade.profiler.chart.json())
    #out2 = hb_out.cpu()

    #assert torch.allclose(out1, out2, atol=1e-6)
