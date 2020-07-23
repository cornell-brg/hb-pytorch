import torch
import sys, os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
import utils # noqa

if __name__ == "__main__":
    args = utils.parse_args()

    inputs = torch.rand(32, 3, 32, 32, requires_grad=True)

    utils.benchmark_module(torch.nn.Conv2d, [inputs], args.backward,
                           3, 4, kernel_size=3)
