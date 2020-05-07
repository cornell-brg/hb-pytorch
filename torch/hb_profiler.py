import torch

class ProfilerStatus:

    def __init__(self):
        self.is_in_ROI = False


profiler_status = ProfilerStatus()

# --------- ExecutionTime Stack Tree -------

class exec_time_Node:

    def __init__(self, func, time, fancy_func=False):
        self.func = func
        self.time = float(time)
        self.children = []
        self.percentage = -1
        if fancy_func:
            func = self.func
            func = func.split("(")[0]
            func = func.split("::")[-1]
            func = "aten::" + func
            self.func = func

    def add_child(self, child):
        assert isinstance(child, exec_time_Node)
        self.children.append(child)

    def __str__(self):
        return "Node(" + self.func + " : " + str(self.time) + ")"


def exec_time_preprocess(data):
    processed = []
    data = data.splitlines()
    roi = ()
    for d in data:
        stack, time = d.split(";")
        if stack == "time_in_roi":
            roi = (stack, time)
        else:
            stack = stack.split("<|>")
            processed.append((stack, time))
    return processed, roi

# recursively construct a stack tree
# the idea is to find len(1) stacks, and construct a node
# anything between two len(1) stacks, or the end, should be
# recursively handled and added as len(1) stack's children
def exec_time_construct_tree_impl(data, parent, fancy_func=False):
    global_idx = 0
    while global_idx < len(data):
        stack, time = data[global_idx]
        assert len(stack) == 1
        node = exec_time_Node(stack[0], time, fancy_func)
        parent.add_child(node)
        lower_level = []
        global_idx += 1
        while global_idx < len(data):
            stack, time = data[global_idx]
            if len(stack) == 1:
                break
            # stack pop front
            lower_level.append((stack[1:], time))
            global_idx += 1
        exec_time_construct_tree_impl(lower_level, node)

# find other time in ROI
def exec_time_add_other(root):
    agg_total = 0.0
    for kid in root.children:
        agg_total += kid.time
    root.children.append(exec_time_Node("other", root.time - agg_total))

# append percentage of ROI to each node
def exec_time_calc_percentage(root, roi_time=None):
    if roi_time is None:
        roi_time = root.time
    root.percentage = root.time / roi_time * 100.0
    for kid in root.children:
        exec_time_calc_percentage(kid, roi_time)

# preorder traversal
def exec_time_print_tree(root, lvl=0, output=None):
    if output is None:
        output = []
    fancy_str = ""
    for i in range(lvl):
        fancy_str += "  "
    fancy_str += "|- "
    fancy_str += str(root)
    output.append(fancy_str)
    for kid in root.children:
        exec_time_print_tree(kid, lvl + 1, output)
    return "\n".join(output)

# wrap everything, return a tree
def exec_time_tree(fancy_func=False):
    data = torch._C._hb_profiler_exec_time_raw_stack()
    data, roi = exec_time_preprocess(data)
    root = exec_time_Node(roi[0], roi[1])
    exec_time_construct_tree_impl(data, root, fancy_func)
    exec_time_add_other(root)
    exec_time_calc_percentage(root)
    return root

# --------- torch.hb_profiler APIs ---------

def enable():
    profiler_status.is_in_ROI = True
    torch._C._hb_profiler_start()

def disable():
    profiler_status.is_in_ROI = False
    torch._C._hb_profiler_end()

def is_in_ROI():
    return profiler_status.is_in_ROI

def add_beacon(signature):
    try:
        torch._C._hb_profiler_chart_add_beacon(signature)
    except AttributeError:
        print("PyTorch is not built with profiling")

def clear_beacon():
    try:
        torch._C._hb_profiler_chart_clear_beacon()
    except AttributeError:
        print("PyTorch is not built with profiling")

def add_waypoint(signature, redispatch):
    try:
        if not torch._C._hb_profiler_route_add_waypoint(signature, redispatch):
            print("PyTorch is not built with redispatching")
    except AttributeError:
        print("PyTorch is not built with profiling")

def set_route_from_json(json):
    try:
        for wp in json:
            add_waypoint(wp['signature'], wp['offload'])
    except (AttributeError, KeyError):
        print("Failed to parse route json or PyTorch is not built with profiling")

def unimpl_print():
    try:
        return torch._C._hb_profiler_unimpl_print()
    except AttributeError:
        print("PyTorch is not built with profiling")

def exec_time_fancy_print(raw_data=None):
    try:
        root = exec_time_tree(fancy_func=True)
        buffer = ""
        for e in (root.children + [root]):
            func = e.func
            time = e.time / 1000.0
            percentage = e.percentage
            buffer += ('{func:30}     {time:.2f} {percentage:.1f}%\n'.format(
                func=func, time=time, percentage=percentage))
        return buffer
    except AttributeError:
        print("PyTorch is not built with profiling")

def exec_time_latex_table(raw_data=None):
    try:
        root = exec_time_tree(fancy_func=True)
        buffer = ""
        header = "\\begin{table}[t]\n" \
                 "\\begin{tabular}{lrr}\n" \
                 "\\toprule\n" \
                 "& \\textbf{Time} & \\textbf{Percent} \\\\\n" \
                 "\\textbf{Kernel} & \\textbf{(s)} & \\textbf{of Total} \\\\ \\midrule\n"
        buffer += header

        for e in (root.children + [root]):
            func = e.func
            func = func.replace("_", "\\_")
            time = e.time / 1000.0
            percentage = e.percentage
            buffer += ('\\textbf{{{func:30}}} &  {time:.2f} & {percentage:.1f}\\% \\\\'.format(
                func=func, time=time, percentage=percentage))

        footer = "\\bottomrule\n" \
                 "\\end{tabular}\n" \
                 "\\label{tbl-plat}\n" \
                 "\\end{table}\n"
        buffer += footer
        return buffer
    except AttributeError:
        print("PyTorch is not built with profiling")

def exec_time_raw_stack():
    try:
        root = exec_time_tree()
        return exec_time_print_tree(root)
    except AttributeError:
        print("PyTorch is not built with profiling")

def route_print():
    try:
        return torch._C._hb_profiler_route_print()
    except AttributeError:
        print("PyTorch is not built with profiling")

def chart_print():
    try:
        return torch._C._hb_profiler_chart_print()
    except AttributeError:
        print("PyTorch is not built with profiling")
