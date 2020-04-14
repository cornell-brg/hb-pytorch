import torch
import torch.nn as nn

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

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, bias=False)
        self.relu_conv1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, bias=False)
        self.relu_conv2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(800, 500, bias=False)
#        self.fc1_weight_coo = load_fc1_weight_coo()
        self.relu_fc1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(500, 10, bias=False)
#        self.fc2_weight_coo = load_fc2_weight_coo()
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu_conv2(x)
        x = self.pool2(x)
#        x = x.view(x.size(0), 50*4*4).t().contiguous()
        x = x.view(x.size(0), 50*4*4)
#        x = torch.sparse.mm(self.fc1_weight_coo, x).t()
        x = self.fc1(x)
        x = self.relu_fc1(x)
#        x = torch.sparse.mm(self.fc2_weight_coo, x).t()
        x = self.fc2(x)
        return x

def test_lenet5_forward_1():
    net = LeNet_5()
    net_hb = LeNet_5().hammerblade()

    data = torch.randn(1, 1, 28, 28)
    data_hb = data.hammerblade()

    output = net(data)
    output_hb = net_hb(data_hb)
    
    print(output)
    print(output_hb)
    
    assert torch.allclose(output, output_hb.cpu())
