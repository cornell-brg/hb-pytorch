import argparse
import sys, os
import torch
import torch.nn as nn
import torch.autograd.profiler as torchprof
import thop

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
import hbutils # noqa

_CYCLE_TIME = 1e-9

def parse_args():
    parser = argparse.ArgumentParser(
           formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--backward', default=False, action='store_true',
                        help="Benchmark the backward.")
    return parser.parse_args()

def _evaluate_op(op, input):
    """
    Accepts an op as a lambda and evaluates the op.

    Returns: tuple with output and execution time in us.
    """
    with torchprof.profile() as prof:
        output = op(input)
    return output, prof.self_cpu_time_total

def benchmark_module(module, inputs, backward=False, *args, **kwargs):
    class Model(nn.Module):
        def __init__(self, *args, **kwargs):
            super(Model, self).__init__()
            self.layer = module(*args, **kwargs)
        def forward(self, x):
            return self.layer(x)

    # Compute FLOPS of this layers
    macs, _ = thop.profile(Model(*args, **kwargs), inputs=(inputs[0],))
    print(macs)

    model = module(*args, **kwargs)

    model_hb = module(*args, **kwargs).hammerblade()
    model_hb.load_state_dict(model.state_dict())

    row_format ="{:>10} | {:>30} | {:>15} | {:>15} | {:>15}"

    print(row_format.format("Layer", "Layer parameters",
                            "Inputs shape", "CPU Time (ms)", "HB Time (ms)"))
    for i in inputs:
        i_hb =  hbutils.init_hb_tensor(i)

        exec_time_cpu, exec_time_hb = 0, 0
        if not backward:
            output_cpu, exec_time_cpu = _evaluate_op(lambda t: model(t), i)
            output_hb, exec_time_hb = _evaluate_op(lambda t: model_hb(t), i_hb)
            assert torch.allclose(output_cpu, output_hb.cpu(), atol=1e-5)
        else:
            forward_output = model(i)
            forward_output_hb = model_hb(i_hb)
            output_cpu, exec_time_cpu = _evaluate_op(lambda t: t.backward(),
                                                     forward_output)
            output_hb, exec_time_hb = _evaluate_op(lambda t: t.backward(),
                                                   forward_output_hb)
            assert torch.allclose(output_cpu, output_hb.cpu(), atol=1e-5)

        exec_time_cpu_ms = "{:6.2f}".format(exec_time_cpu / 1000)
        exec_time_hb_ms = "{:6.2f}".format(exec_time_hb / 1000)
        print(row_format.format(
            module.__name__, str([list(p.shape) for p in model.parameters()]),
            str(list(i.shape)), exec_time_cpu_ms, exec_time_hb_ms))
