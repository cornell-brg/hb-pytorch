"""
Tests on torch.random_
24/01/2021 Zhongyuan Zhao (zz546@cornell.edu)
"""
import torch

def test_random_1():
    x = torch.IntTensor(3072)
    hb_x = x.hammerblade()
    hb_r = hb_x.random_()
    print(hb_r)

def test_random_2():
    x = torch.IntTensor(10, 10)
    hb_x = x.hammerblade()
    hb_r = hb_x.random_()
    print(hb_r)

def test_random_graphsage():
    x = torch.IntTensor(36864)
    hb_x = x.hammerblade()
    hb_r = hb_x.random_()
    print(hb_r)
    
