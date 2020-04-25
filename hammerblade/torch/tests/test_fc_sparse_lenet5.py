import torch
import torch.nn as nn
import hbutils

def load_fc1_weight_coo():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    fc1_weight = weights.get('ip1.weight').cpu()
    print(fc1_weight.size())
    fc1_weight_coo = fc1_weight.to_sparse()
    return fc1_weight_coo

def load_fc2_weight_coo():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    fc2_weight = weights.get('ip2.weight').cpu()
    print(fc2_weight.size())
    fc2_weight_coo = fc2_weight.to_sparse()
    return fc2_weight_coo

def convert_input(x):
    stride = (1, 1)
    kernel_size = (5, 5)
    assert x.dim() == 4
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = torch.flatten(x, start_dim = 4)
    x = x.transpose(1, 3).transpose(1, 2).contiguous()
    x = torch.flatten(x, start_dim = 3)
    x = torch.flatten(x, start_dim = 0, end_dim = 2).t().contiguous()
    return x

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=False)
        self.relu_conv1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, bias=False)
        self.relu_conv2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#        self.fc1 = nn.Linear(800, 500, bias=False)
        self.fc1_weight_coo = load_fc1_weight_coo()
        self.relu_fc1 = nn.ReLU(inplace=False)
#        self.fc2 = nn.Linear(500, 10, bias=False)
        self.fc2_weight_coo = load_fc2_weight_coo()
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), 50*4*4).t().contiguous()
#        x = x.view(x.size(0), 50*4*4)
        x = torch.sparse.mm(self.fc1_weight_coo, x).t()
#        x = self.fc1(x)
        x = self.relu_fc1(x).t()
        x = torch.sparse.mm(self.fc2_weight_coo, x).t()
#        x = self.fc2(x)
        return x

    def forward_conv_to_gemm(self, x):
        x = convert_input(x)
        y = self.conv1.weight.data.view(20, -1).contiguous()
        x = torch.mm(y, x)
        x = x.view(1, 20, 24, 24).contiguous()
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = convert_input(x)
        y = self.conv2.weight.data.view(50, -1).contiguous()
        x = torch.mm(y, x)
        x = x.view(1, 50, 8, 8).contiguous()
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), 50*4*4).t().contiguous()
        x = torch.sparse.mm(self.fc1_weight_coo, x).t()
        x = self.relu_fc1(x).t()
        x = torch.sparse.mm(self.fc2_weight_coo, x).t()
        return x     

def test_lenet5_forward():
    net = LeNet_5()
#    net_hb = LeNet_5().hammerblade()
    
#    net_hb.load_state_dict(net.state_dict())

    data = torch.rand(1, 1, 28, 28)
    
#    data_hb = hbutils.init_hb_tensor(data)
#    data_hb = data.hammerblade()

    output = net.forward(data)
    output1 = net.forward_conv_to_gemm(data)  
#    output_hb = net_hb.forward(data_hb)
#    output_hb1 = net_hb.forward_conv_to_gemm(data_hb)
    
#   print(output1)
#    print(output_hb)
#    print(output_hb)
    assert torch.allclose(output, output1)
#    assert torch.allclose(output_hb.cpu(), output_hb1.cpu())    
#   assert torch.allclose(output1, output_hb.cpu(), atol=1e-7)
