import argparse
import sys, os
import torch
import torch.autograd.profiler as torchprof

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
import hbutils # noqa

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
    model = module(*args, **kwargs)

    model_hb = module(*args, **kwargs).hammerblade()
    model_hb.load_state_dict(model.state_dict())

    print(module.__name__, [list(p.shape) for p in model.parameters()])
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

        print(i.shape, ":", exec_time_cpu, exec_time_hb)
