"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""
import torch

def test_elementwise_mul_1():
    x = torch.ones(1, 10)
    y = torch.ones(1, 10)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_mul_2():
    x = torch.ones(4, 5)
    y = torch.ones(4, 5)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_mul_3():
    x = torch.rand(1, 128)
    y = torch.rand(1, 128)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_mul_4():
    x = torch.rand(16, 32)
    y = torch.rand(16, 32)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_in_place_mul():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    x1.mul_(x2)
    x1_h.mul_(x2_h)
    assert x1_h.device == torch.device("hammerblade")
    x1_h_c = x1_h.cpu()
    assert torch.equal(x1_h_c, x1)

def test_elementwise_div_1():
    x = torch.ones(1, 10)
    y = torch.ones(1, 10)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

def test_elementwise_div_2():
    x = torch.ones(4, 5)
    y = torch.ones(4, 5)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

def test_elementwise_div_3():
    x = torch.rand(1, 128)
    y = torch.rand(1, 128)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

def test_elementwise_div_4():
    x = torch.rand(16, 32)
    y = torch.rand(16, 32)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

def test_elementwise_in_place_div():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    x1.div_(x2)
    x1_h.div_(x2_h)
    assert x1_h.device == torch.device("hammerblade")
    x1_h_c = x1_h.cpu()
    assert torch.allclose(x1_h_c, x1)
