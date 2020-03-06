"""
BRG tests on PyTorch => mainly used to to test HammerBlade device
Jan 31, 2020
Lin Cheng
"""

# we should be able to import torch with no error
def test_import_torch():
    import torch
    assert torch.has_hammerblade


# test if HammerBlade is config'ed
def test_built_with_hammerblade():
    import torch
    assert torch.has_hammerblade
    # we need to manually init hammerblade module
    torch.hammerblade.init()
    assert not torch.hammerblade.has_half


# generic tests on creating HammerBlade tensors
def test_hammerblade_device():
    import torch
    hb = torch.device("hammerblade")
    assert repr(hb) == "device(type='hammerblade')"


def test_hammerblade_empty_path1():
    import torch
    hb = torch.device("hammerblade")
    x = torch.empty((1, 5), device=hb)
    assert x.device == hb
    assert x.type() == 'torch.hammerblade.FloatTensor'
    assert x.is_hammerblade


def test_hammerblade_empty_path2():
    import torch
    hb = torch.device("hammerblade")
    x = torch.empty((1, 5), device="hammerblade")
    assert x.device == hb
    assert x.type() == 'torch.hammerblade.FloatTensor'
    assert x.is_hammerblade


def test_move_between_cpu_and_hammerblade_path1():
    import torch
    cpu = torch.device("cpu")
    hb = torch.device("hammerblade")
    cpu_x = torch.rand(1, 10)
    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.FloatTensor'
    assert not cpu_x.is_hammerblade
    hb_x = cpu_x.hammerblade()
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.FloatTensor'
    assert hb_x.is_hammerblade
    assert torch.equal(hb_x.cpu(), cpu_x)


def test_move_between_cpu_and_hammerblade_path2():
    import torch
    cpu = torch.device("cpu")
    hb = torch.device("hammerblade")
    cpu_x = torch.rand(1, 10)
    assert cpu_x.device == cpu
    assert cpu_x.type() == 'torch.FloatTensor'
    assert not cpu_x.is_hammerblade
    hb_x = cpu_x.to("hammerblade")
    assert hb_x.device == hb
    assert hb_x.type() == 'torch.hammerblade.FloatTensor'
    assert hb_x.is_hammerblade
    assert torch.equal(hb_x.cpu(), cpu_x)
