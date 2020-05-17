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
