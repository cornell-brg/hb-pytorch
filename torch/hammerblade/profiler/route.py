import torch

# --------- torch.hb_profiler.route APIs ---------

def add(signature, redispatch):
    try:
        if not torch._C._hb_profiler_route_add_waypoint(signature, redispatch):
            print("PyTorch is not built with redispatching")
    except AttributeError:
        print("PyTorch is not built with profiling")

def set_route_from_json(json):
    try:
        for wp in json:
            add(wp['signature'], wp['offload'])
    except (AttributeError, KeyError):
        print("Failed to parse route json or PyTorch is not built with profiling")

def json():
    try:
        return torch._C._hb_profiler_route_print()
    except AttributeError:
        print("PyTorch is not built with profiling")

def result_check_enable():
    try:
        torch._C._hb_profiler_route_enable_allclose_check()
    except AttributeError:
        print("PyTorch is not built with profiling")

def result_check_disable():
    try:
        torch._C._hb_profiler_route_disable_allclose_check()
    except AttributeError:
        print("PyTorch is not built with profiling")
