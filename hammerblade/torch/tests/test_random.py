import torch

def test_random_1():
    x = torch.IntTensor(100)
    hb_x = x.hammerblade()
    hb_r = hb_x.random_()
    print(hb_r)

def test_random_2():
    x = torch.IntTensor(10, 10)
    hb_x = x.hammerblade()
    hb_r = hb_x.random_()
    print(hb_r)
    
