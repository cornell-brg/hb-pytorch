"""
Tests on torch.to (copy_hb_to_hb kernel)
03/25/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random

torch.manual_seed(42)
random.seed(42)

def test_torch_to_long_to_int_1():
    x = torch.ones(10, dtype=torch.long)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.int)
    int_h = h.to(torch.int)
    assert int_h.type() == 'torch.hammerblade.IntTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.equal(int_x, int_h.cpu())

def test_torch_to_long_to_int_2():
    x = torch.LongTensor(10).random_(0, 10)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.int)
    int_h = h.to(torch.int)
    assert int_h.type() == 'torch.hammerblade.IntTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.equal(int_x, int_h.cpu())

def test_torch_to_long_to_int_3():
    x = torch.LongTensor(64).random_(-2147483648, 2147483647)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.int)
    int_h = h.to(torch.int)
    assert int_h.type() == 'torch.hammerblade.IntTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.equal(int_x, int_h.cpu())

def test_torch_to_long_to_float_1():
    x = torch.ones(10, dtype=torch.long)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.float)
    int_h = h.to(torch.float)
    assert int_h.type() == 'torch.hammerblade.FloatTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.allclose(int_x, int_h.cpu())

def test_torch_to_long_to_double_1():
    x = torch.ones(10, dtype=torch.long)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.double)
    int_h = h.to(torch.double)
    assert int_h.type() == 'torch.hammerblade.DoubleTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.allclose(int_x, int_h.cpu())

def test_torch_to_int_to_long_1():
    x = torch.ones(10, dtype=torch.int)
    assert x.type() == 'torch.IntTensor'
    h = x.hammerblade()
    long_x = x.to(torch.long)
    long_h = h.to(torch.long)
    assert long_h.type() == 'torch.hammerblade.LongTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.equal(long_x, long_h.cpu())

def test_torch_to_int_to_float_1():
    x = torch.ones(10, dtype=torch.int)
    assert x.type() == 'torch.IntTensor'
    h = x.hammerblade()
    long_x = x.to(torch.float)
    long_h = h.to(torch.float)
    assert long_h.type() == 'torch.hammerblade.FloatTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())

def test_torch_to_int_to_double_1():
    x = torch.ones(10, dtype=torch.int)
    assert x.type() == 'torch.IntTensor'
    h = x.hammerblade()
    long_x = x.to(torch.double)
    long_h = h.to(torch.double)
    assert long_h.type() == 'torch.hammerblade.DoubleTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())

def test_torch_to_float_to_int_1():
    x = torch.randn(10, dtype=torch.float)
    assert x.type() == 'torch.FloatTensor'
    h = x.hammerblade()
    long_x = x.to(torch.int)
    long_h = h.to(torch.int)
    assert long_h.type() == 'torch.hammerblade.IntTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())

def test_torch_to_float_to_long_1():
    x = torch.randn(10, dtype=torch.float)
    assert x.type() == 'torch.FloatTensor'
    h = x.hammerblade()
    long_x = x.to(torch.long)
    long_h = h.to(torch.long)
    assert long_h.type() == 'torch.hammerblade.LongTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())

def test_torch_to_float_to_double_1():
    x = torch.randn(10, dtype=torch.float)
    assert x.type() == 'torch.FloatTensor'
    h = x.hammerblade()
    long_x = x.to(torch.double)
    long_h = h.to(torch.double)
    assert long_h.type() == 'torch.hammerblade.DoubleTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())

def test_torch_to_double_to_int_1():
    x = torch.randn(10, dtype=torch.double)
    assert x.type() == 'torch.DoubleTensor'
    h = x.hammerblade()
    long_x = x.to(torch.int)
    long_h = h.to(torch.int)
    assert long_h.type() == 'torch.hammerblade.IntTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())

def test_torch_to_double_to_long_1():
    x = torch.randn(10, dtype=torch.double)
    assert x.type() == 'torch.DoubleTensor'
    h = x.hammerblade()
    long_x = x.to(torch.long)
    long_h = h.to(torch.long)
    assert long_h.type() == 'torch.hammerblade.LongTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())

def test_torch_to_double_to_float_1():
    x = torch.randn(10, dtype=torch.double)
    assert x.type() == 'torch.DoubleTensor'
    h = x.hammerblade()
    long_x = x.to(torch.float)
    long_h = h.to(torch.float)
    assert long_h.type() == 'torch.hammerblade.FloatTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.allclose(long_x, long_h.cpu())
