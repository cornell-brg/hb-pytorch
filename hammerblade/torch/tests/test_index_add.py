import torch

def _test_index_add(dim, t1, t2, index):
    t1_h = t1.hammerblade()
    t2_h = t2.hammerblade()
    index_h = index.hammerblade()

    out_h = t1_h.index_add(dim, index_h, t2_h)
    out = t1.index_add(dim, index, t2)
    print(out)
    print(out_h)
    #abs_diff = torch.abs(out - out_h.cpu())
    #nonzero = torch.nonzero(abs_diff)
    #print(nonzero)
    #print("#nonzero: ", nonzero.size())
    #print("sum abs difference: ", torch.sum(abs_diff))
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out, out_h.cpu())


def test_index_add_1():
    dim = 0
    t1 = torch.ones(5, 3)
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    index = torch.tensor([3, 2, 1])
    _test_index_add(dim, t1, t2, index)

def test_index_add_2():
    dim = 1
    t1 = torch.ones(5, 3)
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [0, 0, 0]], dtype=torch.float)
    index = torch.tensor([2, 0, 1])
    _test_index_add(dim, t1, t2, index)

def test_index_add_3():
    dim = 0
    t1 = torch.tensor([[[[1, 1], [2, 2], [3, 3]], [[11, 11], [22, 22], [33, 33]]],
                       [[[4, 4], [5, 5], [6, 6]], [[44, 44], [55, 55], [66, 66]]],
                       [[[7, 7], [8, 8], [9, 9]], [[77, 77], [88, 88], [99, 99]]]], dtype=torch.float)
    print(t1)
    print("size: ", t1.size()) # 3, 2, 3, 2
    print("strides: ", t1.stride())

    t2 = torch.rand([2, 2, 3, 2])
    print(t2)
    index = torch.tensor([1, 0])
    print(index)
    _test_index_add(dim, t1, t2, index)

def test_index_add_4():
    dim = 1
    t1 = torch.tensor([[[[1, 1], [2, 2], [3, 3]], [[11, 11], [22, 22], [33, 33]]],
                       [[[4, 4], [5, 5], [6, 6]], [[44, 44], [55, 55], [66, 66]]],
                       [[[7, 7], [8, 8], [9, 9]], [[77, 77], [88, 88], [99, 99]]]], dtype=torch.float)
    print(t1)
    print("size: ", t1.size()) # 3, 2, 3, 2
    print("strides: ", t1.stride())

    t2 = torch.rand([3, 1, 3, 2])
    print(t2)
    index = torch.tensor([0])
    print(index)
    _test_index_add(dim, t1, t2, index)

def test_index_add_5():
    dim = 2
    t1 = torch.tensor([[[[1, 1], [2, 2], [3, 3]], [[11, 11], [22, 22], [33, 33]]],
                       [[[4, 4], [5, 5], [6, 6]], [[44, 44], [55, 55], [66, 66]]],
                       [[[7, 7], [8, 8], [9, 9]], [[77, 77], [88, 88], [99, 99]]]], dtype=torch.float)
    print(t1)
    print("size: ", t1.size()) # 3, 2, 3, 2
    print("strides: ", t1.stride())

    t2 = torch.rand([3, 2, 5, 2])
    print(t2)
    index = torch.tensor([0, 0, 1, 2, 2])
    print(index)
    _test_index_add(dim, t1, t2, index)

def test_index_add_6():
    dim = 3
    t1 = torch.tensor([[[[1, 1], [2, 2], [3, 3]], [[11, 11], [22, 22], [33, 33]]],
                       [[[4, 4], [5, 5], [6, 6]], [[44, 44], [55, 55], [66, 66]]],
                       [[[7, 7], [8, 8], [9, 9]], [[77, 77], [88, 88], [99, 99]]]], dtype=torch.float)
    print(t1)
    print("size: ", t1.size()) # 3, 2, 3, 2
    print("strides: ", t1.stride())

    t2 = torch.rand([3, 2, 3, 1])
    print(t2)
    index = torch.tensor([0])
    print(index)
    _test_index_add(dim, t1, t2, index)

def test_index_add_7():
    dim = 4
    t1 = torch.rand([11, 3, 7, 5, 20, 2])
    t2 = torch.rand([11, 3, 7, 5, 10, 2])
    index = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    _test_index_add(dim, t1, t2, index)

def test_index_add_8():
    dim = 4
    t1 = torch.rand([11, 3, 1, 5, 20, 2])
    t2 = torch.rand([11, 3, 1, 5, 10, 2])
    index = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    _test_index_add(dim, t1, t2, index)

def test_index_add_9():
    dim = 2
    t1 = torch.rand([350, 36, 1500])
    t2 = torch.rand([350, 36, 140])
    index = torch.ones(140, dtype=torch.int64)
    print(index)
    _test_index_add(dim, t1, t2, index)

def test_index_add_10():
    dim = 2
    t1 = torch.rand([350, 36, 1500])
    t2 = torch.rand([350, 36, 140])
    index = torch.randint(0, 1500, (140,))
    print(index)
    _test_index_add(dim, t1, t2, index)

def test_index_add_11():
    dim = 2
    t1 = torch.rand([350, 36, 1500])
    t2 = torch.rand([350, 36, 140])
    index = torch.arange(0, 140, dtype=torch.int64)
    print(index)
    _test_index_add(dim, t1, t2, index)

def test_index_add_12():
    dim = 1
    count = 100
    t1 = torch.ones([45, 10])
    t2 = torch.ones([45, count])
    index = torch.zeros(count, dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)

def test_index_add_13():
    dim = 0
    count = 10000
    t1 = torch.ones([10, 2])
    t2 = torch.ones([count, 2])
    index = torch.zeros(count, dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)

def test_index_add_14():
    dim = 0
    t1 = torch.zeros(5)
    t2 = torch.rand(10)
    index = torch.tensor([0, 0, 3, 3, 3, 0, 4, 4, 4, 4], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)

def test_index_add_15():
    dim = 0
    t1 = torch.tensor([42.])
    t2 = torch.rand([5])
    index = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)

def test_index_add_16():
    dim = 2
    t1 = torch.rand([1, 7, 11, 1])
    t2 = torch.rand([1, 7, 7, 1])
    index = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)

def test_index_add_17():
    dim = 1
    t1 = torch.rand([1, 11, 1])
    t2 = torch.rand([1, 7, 1])
    index = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)
