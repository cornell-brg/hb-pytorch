import torch
import torch.nn as nn
import torch.nn.functional as F

def load_conv1_sparse_weight_reshape():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv1_weight = weights.get('conv1.weight').cpu()
    conv1_weight = conv1_weight.view(20, -1).contiguous()
    conv1_weight_coo = conv1_weight.to_sparse()
    return conv1_weight_coo

def load_conv1_sparse_weight():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv1_weight = weights.get('conv1.weight').cpu()
    conv1_weight_coo = conv1_weight.to_sparse().hammerblade()
    print(conv1_weight_coo._nnz())
    return conv1_weight_coo    

def load_conv2_sparse_weight_reshape():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv2_weight = weights.get('conv2.weight').cpu()
    conv2_weight = conv2_weight.view(50, -1).contiguous()
    conv2_weight_coo = conv2_weight.to_sparse()
    return conv2_weight_coo

def load_conv2_sparse_weight():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    conv2_weight = weights.get('conv2.weight').cpu()
    conv2_weight_coo = conv2_weight.to_sparse().hammerblade()
    print(conv2_weight_coo._nnz())
    return conv2_weight_coo

def load_fc1_weight_coo():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    fc1_weight = weights.get('ip1.weight').cpu()
    fc1_weight_coo = fc1_weight.to_sparse()
    print(fc1_weight_coo._nnz())
    return fc1_weight_coo

def load_fc2_weight_coo():
    model = torch.load("LeNet_5.prune.only.fc.pth.tar", map_location='cpu')
    weights = model.get('state_dict')
    fc2_weight = weights.get('ip2.weight').cpu()
    fc2_weight_coo = fc2_weight.to_sparse()
    print(fc2_weight_coo._nnz())
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
        self.conv1_sparse_weight_reshape = load_conv1_sparse_weight_reshape()
        self.conv1_sparse_weight = load_conv1_sparse_weight()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=False)
        self.relu_conv1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_sparse_weight_reshape = load_conv2_sparse_weight_reshape()
        self.conv2_sparse_weight = load_conv2_sparse_weight()
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
    
    def forward_sparse_conv_to_gemm(self, x):
        x = convert_input(x)
        y = self.conv1_sparse_weight_reshape
        x = torch.sparse.mm(y, x)
        x = x.view(1, 20, 24, 24).contiguous()
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = convert_input(x)
        y = self.conv2_sparse_weight_reshape
        x = torch.sparse.mm(y, x)
        x = x.view(1, 50, 8, 8).contiguous()
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), 50*4*4).t().contiguous()
        x = torch.sparse.mm(self.fc1_weight_coo, x).t()
        x = self.relu_fc1(x).t()
        x = torch.sparse.mm(self.fc2_weight_coo, x).t()
        return x

    def forward_sparse_conv_for_hb(self, x):
        y = self.conv1_sparse_weight
        x = F.conv2d(x, y, bias = None, stride = 1, padding = 0, dilation = 1)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        y = self.conv2_sparse_weight
        x = F.conv2d(x, y, bias = None, stride = 1, padding = 0, dilation = 1)
        x = self.relu_conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), 50*4*4).t().contiguous()
        x = torch.sparse.mm(self.fc1_weight_coo.hammerblade(), x).t()
        x = self.relu_fc1(x).t()
        x = torch.sparse.mm(self.fc2_weight_coo.hammerblade(), x).t()
        return x

def test_lenet5_forward():
    net = LeNet_5()
    
#    net_hb.load_state_dict(net.state_dict())
    torch.hammerblade.profiler.enable()
    data = torch.rand(1, 1, 28, 28)
    data_hb = data.hammerblade()
    hb_result = net.forward_sparse_conv_for_hb(data_hb)
    torch.hammerblade.profiler.disable()

    output = net.forward(data)
    output1 = net.forward_conv_to_gemm(data)  
    cpu_result  = net.forward_sparse_conv_to_gemm(data)
    assert torch.allclose(output, output1)   
    assert torch.allclose(cpu_result, hb_result.cpu())
