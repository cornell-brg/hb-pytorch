import torch
import sys, os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
import utils # noqa

if __name__ == "__main__":
    args = utils.parse_args()

    log = ""

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

    #log += utils.benchmark_module(torch.nn.Conv2d, inputs, args.backward,
    #                              in_channels, out_channels, kernel_size)

    in_channels = 3
    out_channels = 16
    kernel_size = 3
    inputs = [
        torch.rand(1, in_channels, 8, 8, requires_grad=True),
        torch.rand(1, in_channels, 32, 32, requires_grad=True),
        torch.rand(1, in_channels, 64, 64, requires_grad=True),
        torch.rand(8, in_channels, 32, 32, requires_grad=True),
        #torch.rand(8, in_channels, 64, 64, requires_grad=True),
        #torch.rand(16, in_channels, 32, 32, requires_grad=True),
        #torch.rand(16, in_channels, 64, 64, requires_grad=True),
        #torch.rand(32, in_channels, 32, 32, requires_grad=True),
        #torch.rand(32, in_channels, 64, 64, requires_grad=True),
        #torch.rand(64, in_channels, 32, 32, requires_grad=True),
        #torch.rand(64, in_channels, 64, 64, requires_grad=True),
    ]

    log += utils.benchmark_module(torch.nn.Conv2d, inputs, args.backward,
                                  in_channels, out_channels, kernel_size)

    print(log)
