import torch.hammerblade.profiler as hbprof
import argparse
import utils
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
import hbutils

if __name__ == "__main__":
    args = utils.parse_args()

    inputs = torch.rand(32, 3, 32, 32, requires_grad=True)
    kernel = torch.rand(8, 3, 3, 3, requires_grad=True)

    output = F.conv2d(inputs, kernel)
    grad = torch.rand(output.shape)

    if args.hammerblade:
        inputs = hbutils.init_hb_tensor(inputs)
        kernel = hbutils.init_hb_tensor(kernel)
        grad = grad.hammerblade()

    output = F.conv2d(inputs, kernel)

    hbprof.enable()
    output.backward(grad)
    hbprof.disable()
    print(hbprof.stats(key=['ExecTime']))
