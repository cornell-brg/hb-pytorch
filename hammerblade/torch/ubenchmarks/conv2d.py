import torch
import torch.nn.functional as F
import torch.hammerblade.profiler as hbprof
import argparse
import utils
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
import hbutils

if __name__ == "__main__":
    args = utils.parse_args()

    inputs = torch.rand(32, 16, 32, 32, requires_grad=True)
    kernel = torch.rand(32, 16, 3, 3, requires_grad=True)

    output_ref = F.conv2d(inputs, kernel)

    if args.hammerblade:
        inputs = hbutils.init_hb_tensor(inputs)
        kernel = hbutils.init_hb_tensor(kernel)

    hbprof.enable()
    output = F.conv2d(inputs, kernel)
    hbprof.disable()
    print(hbprof.stats(key=['ExecTime']))

    assert torch.allclose(output_ref, output.cpu())
