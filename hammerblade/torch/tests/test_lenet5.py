"""
Test on a small CNN
03/16/2020 Bandhav Veluri
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

# Network
class LeNet5(nn.Module):
  """
  LeNet-5

  https://cs.nyu.edu/~yann/2010f-G22-2565-001/diglib/lecun-98.pdf
  (Page 7)
  """
  def __init__(self):
    super(LeNet5, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(1, 6, kernel_size=(5,5)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2,2), stride=2),
      nn.Conv2d(6, 16, kernel_size=(5,5)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2,2), stride=2),
      nn.Conv2d(16, 120, kernel_size=(5,5)),
    )

    self.fc = nn.Sequential(
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, 10),
      nn.LogSoftmax(dim=-1),
    )

  def forward(self, data):
    x = self.conv(data)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x

# Train routine
def train(net, loader, epochs, optimizer, loss_func):
  print('Training {} for {} epoch(s)...\n'.format(type(net).__name__, epochs))
  for epoch in range(epochs):
    losses = []

    for batch_idx, (data, labels) in enumerate(loader, 0):
      batch_size = len(data)
      optimizer.zero_grad()
      outputs = net(data)
      loss = loss_func(outputs, labels)
      losses.append(loss.item())
      loss.backward()
      optimizer.step()

      if (batch_idx % 1000) == 0:
        print('epoch {} : [{}/{} ({:.0f}%)]\tLoss={:.6f}'.format(
          epoch, batch_idx*batch_size, len(loader.dataset),
          100. * (batch_idx/len(loader)), loss.item()
        ))

    print('epoch {} : Average Loss={:.6f}\n'.format(
      epoch, np.mean(losses)
    ))

# Test routine
@pytest.mark.skip(reason = "Not a pytest test, this is CNN test routine.")
@torch.no_grad()
def test(net, loader, loss_func):
  test_loss = 0.0
  num_correct = 0

  for batch_idx, (data, labels) in enumerate(loader, 0):
    output = net(data)
    loss = loss_func(output, labels)
    pred = output.max(1)[1]
    num_correct += pred.eq(labels.view_as(pred)).sum().item()

  test_loss /= len(loader.dataset)
  test_accuracy = 100. * (num_correct / len(loader.dataset))

  print('Test set: Average loss={:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, num_correct, len(loader.dataset), test_accuracy
  ))

def test_lenet5_forward_1():
    net = LeNet5()
    net_hb = LeNet5().hammerblade()

    data = torch.rand(1, 1, 32, 32)
    data_hb = data.hammerblade()

    output = net.forward(data)
    output_hb = net_hb.forward(data_hb)

    assert torch.allclose(output, output_hb.cpu())

@pytest.mark.skip(reason = "Not implented for HB yet.")
def test_lenet5_mnist():
    """
    Tests one epoch of train and test on LeNet-5 with MNIST dataset.
    """
    # Model
    BATCH_SIZE = 32
    LEARNING_RATE = 0.02
    MOMENTUM = 0.9
    EPOCHS = 1
    net = LeNet5()
    optimizer = torch.optim.SGD(
      net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
    )
    loss_func = nn.CrossEntropyLoss()

    # Data
    transforms = torchvision.transforms.Compose([
      torchvision.transforms.Resize((32,32)),
      torchvision.transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
      root='./data/mnist', train=True, download=True, transform=transforms
    )
    trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.MNIST(
      root='./data/mnist', train=False, download=True, transform=transforms
    )
    testloader = torch.utils.data.DataLoader(
      testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Train
    train(net, trainloader, EPOCHS, optimizer, loss_func)

    # Test
    test(net, testloader, loss_func)
