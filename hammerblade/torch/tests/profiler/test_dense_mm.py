import torch
import torch.nn.functional as F
import json

def test_dense_mm():
    torch.hammerblade.init()
    m1 = torch.rand(10, 8)
    m2 = torch.rand(8, 6)
    
    torch.hammerblade.profiler.chart.add("at::Tensor at::CPUType::{anonymous}::mm(const at::Tensor&, const at::Tensor&)")
    
   # with open('conv2_spmm.json') as route:
   #     data = json.load(route)
   # torch.hammerblade.profiler.route.set_route_from_json(data)

    torch.hammerblade.profiler.enable()
#    cpu_i = im2col(di, 5, 5)
    m = torch.mm(m1, m2)

#    hb_i = cpu_i.hammerblade()
#    hb_sw = cpu_sw.hammerblade()
#    hb_out = torch.sparse.mm(hb_sw, hb_i)
    torch.hammerblade.profiler.disable()
#    print(torch.hammerblade.profiler.exec_time.raw_stack())
    print(torch.hammerblade.profiler.chart.json())

#    out2 = hb_out.cpu()
#    assert torch.allclose(cpu_out, out2, atol=1e-6)
