import torch
import torch.nn as nn

def convert_input_dense(x):
    stride = (1, 1)
    kernel_size = (5, 5)
    assert x.dim() == 4
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = torch.flatten(x, start_dim = 4)
    x = x.transpose(1, 3).transpose(1, 2)
    x = torch.flatten(x, start_dim = 3)
    x = torch.flatten(x, start_dim = 0, end_dim = 2).t().contiguous()

def convert_weight_sparse(x):
    x = torch.flatten(x, start_dim = 1)  
   
    

class TestNet(nn.Module):
    def __init__(self, x, y):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size = 5, stride = 1, bias = False)
        self.convert_input = convert_input_dense()
        self.convert_weight = convert_weight_sparse()
        return

    def conv_layer(self, x):
        x = self.conv1(x)
        return x
    
    def conv_to_gemm(self, x, y):
        x = self.convert_input(x)
        y = self.convert_weight(y)
        z = torch.sparse.mm(y, x).t()
        return z

def test_conv_to_gemm():
    inputs = torch.ones(1, 1, 28, 28)
    weight = torch.ones(20, 1, 5, 5)
    net = TestNet(inputs, weight)
    module.state_dict().keys() 
    output = net.conv_layer(inputs)
    output1 = net.conv_to_gemm(inputs, weight)
    
    print(output)
    print(output1)
    assert torch.allclose(output, output1)
        
        
        
