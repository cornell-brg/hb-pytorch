"""
Tests on torch.index
02/02/2021 Zhongyuan Zhao (zz546@cornell.edu)
"""
import torch

def test_indexing_1D_array():
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).float()
    hb_x = x.hammerblade()

    #Index single element:
    z1 = x[2]  #Outpt[3]

    #slices or strides a piece of tensor from input tensor
    z2 = x[2:5] #Output[3, 4, 5]
#    z3 = x[2:6:2] #Output[3, 5]

    #Index multiple elements using a 1D tensor as the indexer
    y1 = torch.tensor([1, 4, 6])
    z4 = x[y1] #Output[2, 5, 7]

    #Index multiple elememts using a 2D tensor as the indexer
    y2 = torch.tensor([[0, 1, 2], [2, 4, 6]])
    z5 = x[y2] #Output [[1, 2, 3], [3, 5, 7]]

    hb_z1 = hb_x[2]
    hb_z2 = hb_x[2:5]
#    hb_z3 = hb_x[2:6:2]
    
    hb_y1 = y1.hammerblade()
    hb_y2 = y2.hammerblade()

    hb_z4 = hb_x[hb_y1]
    hb_z5 = hb_x[hb_y2]

    assert torch.allclose(z1, hb_z1.cpu())
    assert torch.allclose(z2, hb_z2.cpu())
#    assert torch.allclose(z3, hb_z3.cpu())
    assert torch.allclose(z4, hb_z4.cpu())
    assert torch.allclose(z5, hb_z5.cpu())

def test_indexing_2D_array():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
    hb_x = x.hammerblade()
     
    #Index single element
    r1 = x[1, 2] # Output [6]

    #Slicing rows or columns
    r2 = x[0:2]  # Output [[1, 2, 3], [4, 5, 6]]
#    r3 = x[:, 0:2] #Output [[1, 2], [4, 5], [7, 8]]
#    print(r3)

    #Index rows or columns using 1D indexer
    y1 = torch.tensor([0, 2])
    r4 = x[y1] #Output [[1, 2, 3], [7, 8, 9]]
    r5 = x[:, y1] #Output [[1, 3], [4, 6], [7, 9]]

    #Index rows or columns using 2D indexer
    y2 = torch.tensor([[0, 2], [1, 2]])
    r6 = x[y2] #Output [[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [7, 8, 9]]]

    #Index single emelents using two 1D indexers
    y3 = torch.tensor([0, 1]) 
    z3 = torch.tensor([1, 2])
    r7 = x[y3, z3] #Output [2, 6]
    print(r7)

    #Index single elements using two 2D indexers
    y4 = torch.tensor([[0, 1], [0, 1]])
    z4 = torch.tensor([[1, 2], [1, 2]])
    r8 = x[y4, z4] #Output [[2, 6], [2, 6]]
    print(r8)

    hb_r1 = hb_x[1, 2]
    hb_r2 = hb_x[0:2]
 #   hb_r3 = hb_x[:, 0:2]

    hb_y1 = y1.hammerblade()
    hb_y2 = y2.hammerblade()
    hb_y3 = y3.hammerblade()
    hb_z3 = z3.hammerblade()
    hb_y4 = y4.hammerblade()
    hb_z4 = z4.hammerblade()

    hb_r4 = hb_x[hb_y1]
    hb_r5 = hb_x[:, y1]
    hb_r6 = hb_x[hb_y2]
    hb_r7 = hb_x[hb_y3, hb_z3]
    print(hb_r7)
    hb_r8 = hb_x[hb_y4, hb_z4]
    print(hb_r8)

    assert torch.allclose(r1, hb_r1.cpu())
    assert torch.allclose(r2, hb_r2.cpu())
#    assert torch.allclose(r3, hb_r3.cpu())
    assert torch.allclose(r4, hb_r4.cpu())
    assert torch.allclose(r5, hb_r5.cpu())
    assert torch.allclose(r6, hb_r6.cpu())
    assert torch.allclose(r7, hb_r7.cpu())
    assert torch.allclose(r8, hb_r8.cpu())
