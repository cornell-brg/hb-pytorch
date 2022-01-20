import torch

def _test_index_add(dim, t1, t2, index):
    t1_h = t1.hammerblade()
    t2_h = t2.hammerblade()
    index_h = index.hammerblade()

    out_h = t1_h.index_add(dim, index_h, t2_h)
    out = t1.index_add(dim, index, t2)
    #print(out)
    #print(out_h)
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
    print("*** test_index_add_1 done!")

def test_index_add_2():
    dim = 1
    t1 = torch.ones(5, 3)
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [0, 0, 0]], dtype=torch.float)
    index = torch.tensor([2, 0, 1])
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_2 done!")

# test index_add operation for different dimmensions
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
    print("*** test_index_add_3 done!")

# test index_add operation for different dimmensions
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
    print("*** test_index_add_4 done!")

# test index_add operation for different dimmensions
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
    print("*** test_index_add_5 done!")

# test index_add operation for different dimmensions
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
    print("*** test_index_add_6 done!")

# check with 1 dimensional layer inside
def test_index_add_7():
    dim = 1
    t1 = torch.rand([14, 7, 1, 20])
    t2 = torch.rand([14, 6, 1, 20])
    index = torch.tensor([1, 6, 2, 5, 3, 4])
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_7 done!")

# test write to same memory
def test_index_add_8():
    dim = 0
    count = 499
    t1 = torch.ones([10, 2])
    t2 = torch.ones([count, 2])
    index = torch.zeros(count, dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_8 done!")

# check 1D input
def test_index_add_9():
    dim = 0
    t1 = torch.zeros(5)
    t2 = torch.rand(10)
    index = torch.tensor([0, 0, 3, 3, 3, 0, 4, 4, 4, 4], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_9 done!")

# check 0D input
def test_index_add_10():
    dim = 0
    t1 = torch.tensor([42.])
    t2 = torch.rand([5])
    index = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_10 done!")

# check 1 dimensional layers on outside
def test_index_add_11():
    dim = 2
    t1 = torch.rand([1, 7, 11, 1])
    t2 = torch.rand([1, 7, 7, 1])
    index = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_11 done!")

# check 1 dimensional layers on outside
def test_index_add_12():
    dim = 1
    t1 = torch.rand([1, 11, 1])
    t2 = torch.rand([1, 7, 1])
    index = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_12 done!")

# check with slices greater than number of cores
def test_index_add_13():
    dim = 1
    t1 = torch.rand([15, 5, 10])
    t2 = torch.rand([15, 18, 10])
    index = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.int64)
    _test_index_add(dim, t1, t2, index)
    print("*** test_index_add_13 done!")


# tests with realistic size
def _profiling(add_size, recurrence):
    dim = 0
    t1 = torch.rand([3000, 10])
    t2 = torch.rand([add_size, 10])
    recurrence_size = add_size // recurrence
    rest = add_size % recurrence
    index = torch.arange(0, recurrence_size+rest, dtype=torch.int64)
    t_partial = torch.arange(0, recurrence_size, dtype=torch.int64)
    for i in range(1, recurrence):
        index = torch.cat((t_partial, index))
    _test_index_add(dim, t1, t2, index)

# def test_real_10():
#     _profiling(10, 1)
# def test_real_100():
#     _profiling(100, 1)
# def test_real_1000():
#     _profiling(1000, 1)
# def test_real_2000():
#     _profiling(2000, 1)

# def test_rec_3():
#     _profiling(1000, 3)
# def test_rec_6():
#     _profiling(1000, 6)
# def test_rec_9():
#     _profiling(1000, 9)
