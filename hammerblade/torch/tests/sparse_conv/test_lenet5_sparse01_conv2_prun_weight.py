import torch
import torch.nn.functional as F

m01 = torch.nn.Threshold(0.9, 0)

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
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv2_weight = weights.get('conv2.weight').cpu()
    conv2_weight_coo = conv2_weight.to_sparse().hammerblade()
    return conv2_weight_coo

def load_conv2_sparse_weight_reshape():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv2_weight = weights.get('conv2.weight').cpu()
    conv2_weight = conv2_weight.view(50, -1).contiguous()
    conv2_weight_coo = conv2_weight.to_sparse()
    return conv2_weight_coo

def test_lenet5_sparse01_conv2():

    di = torch.rand(1, 20, 12, 12)

    cpu_i = convert_dense_input(di, 5, 5)
    cpu_sw = load_conv2_sparse_weight_reshape()
    cpu_out = torch.sparse.mm(cpu_sw, cpu_i)
    out1 = cpu_out.view(1, 50, 8, 8)

    hb_i = di.hammerblade()
    hb_sw = load_conv2_sparse_weight()
    hb_out = F.conv2d(hb_i, hb_sw, bias = None, stride = 1, padding = 0, dilation = 1)
    out2 = hb_out.cpu()

    assert torch.allclose(out1, out2)
