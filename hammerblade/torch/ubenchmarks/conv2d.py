import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
import ubench  # noqa

if __name__ == "__main__":
    args = ubench.parse_args()

    log = ubench.header()

    in_channels = 3
    out_channels = 8
    kernel_size = 3
    inputs = [
        torch.rand(1, in_channels, 8, 8, requires_grad=True),
        torch.rand(1, in_channels, 32, 32, requires_grad=True),
        torch.rand(1, in_channels, 64, 64, requires_grad=True),
        torch.rand(8, in_channels, 32, 32, requires_grad=True),
        torch.rand(8, in_channels, 64, 64, requires_grad=True),
        torch.rand(16, in_channels, 32, 32, requires_grad=True),
        torch.rand(16, in_channels, 64, 64, requires_grad=True),
        torch.rand(32, in_channels, 32, 32, requires_grad=True),
        torch.rand(32, in_channels, 64, 64, requires_grad=True),
        torch.rand(64, in_channels, 32, 32, requires_grad=True),
        torch.rand(64, in_channels, 64, 64, requires_grad=True),
    ]

    log += ubench.benchmark_module(args.backward, inputs, torch.nn.Conv2d,
                                   in_channels, out_channels, kernel_size)

    in_channels = 8
    out_channels = 16
    kernel_size = 3
    inputs = [
        torch.rand(1, in_channels, 8, 8, requires_grad=True),
        torch.rand(1, in_channels, 32, 32, requires_grad=True),
        torch.rand(1, in_channels, 64, 64, requires_grad=True),
        torch.rand(8, in_channels, 32, 32, requires_grad=True),
        torch.rand(8, in_channels, 64, 64, requires_grad=True),
        torch.rand(16, in_channels, 32, 32, requires_grad=True),
        torch.rand(16, in_channels, 64, 64, requires_grad=True),
        torch.rand(32, in_channels, 32, 32, requires_grad=True),
        torch.rand(32, in_channels, 64, 64, requires_grad=True),
        torch.rand(64, in_channels, 32, 32, requires_grad=True),
        torch.rand(64, in_channels, 64, 64, requires_grad=True),
    ]

    log += ubench.benchmark_module(args.backward, inputs, torch.nn.Conv2d,
                                   in_channels, out_channels, kernel_size)

    print(log)
