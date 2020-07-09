import torch
import torch.nn.functional as F

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

def load_conv2_sparse_weight_reshape():
    model = torch.load("LeNet_5.prune.conv.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv2_weight = weights.get('conv2.weight').cpu()
    conv2_weight = conv2_weight.view(50, -1).contiguous()
    conv2_weight_coo = conv2_weight.to_sparse()
    return conv2_weight_coo

def test_lenet5_sparse01_conv2():

    di = torch.rand(1, 20, 12, 12)
    
    cpu_sw = load_conv2_sparse_weight_reshape()
   # torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::_sparse_mm(const at::Tensor&, const at::Tensor&)")
    route = """[
    {
        "offload": true,
        "signature": "at::Tensor at::SparseCPUType::{anonymous}::_sparse_mm(const at::Tensor&, const at::Tensor&)"
    }
    ]
    """
    data = json.loads(route)
    torch.hammerblade.profiler.route.set_route_from_json(data)

    torch.hammerblade.profiler.enable()
    cpu_i = convert_dense_input(di, 5, 5)
    cpu_out = torch.sparse.mm(cpu_sw, cpu_i)

#    hb_i = cpu_i.hammerblade()
#    hb_sw = cpu_sw.hammerblade()
#    hb_out = torch.sparse.mm(hb_sw, hb_i)
    torch.hammerblade.profiler.disable()
   # print(torch.hammerblade.profiler.exec_time.raw_stack())
   # print(torch.hammerblade.profiler.chart.json())

#    out2 = hb_out.cpu()

#    assert torch.allclose(cpu_out, out2, atol=1e-6)
