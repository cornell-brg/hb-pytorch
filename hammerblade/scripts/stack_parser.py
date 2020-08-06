"""
This is a number of functions to parse the execution stack log
It's almost the same as torch/hammerblade/profiler/exec_time.py
But this is meant for a single ATen Operator's log
06/04/2020 Lin Cheng (lc873@cornell.edu)
"""

class exec_time_Node:

    def __init__(self, func, time, fancy_func=False):
        self.func = func
        self.time = float(time)
        self.children = []
        self.percentage = -1
        if fancy_func and not self.func.startswith("@"):
            func = self.func
            func = func.split("(")[0]
            func = func.split("::")[-1]
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
        if len(d) == 0:
            continue
        if d.startswith("#TOP_LEVEL_FUNC_END#__"):
            continue
        if d.startswith("#TOP_LEVEL_FUNC#__"):
            signature = d[len("#TOP_LEVEL_FUNC#__"):]
            continue
        stack, time = d.split(";")
        if stack == signature:
            roi = ("time_in_roi", time)
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
        exec_time_construct_tree_impl(lower_level, node, fancy_func=fancy_func)

# "Trim" the exec_time tree -> replace offload_kernel time
# with simulated time
def exec_time_apply_trim(root):
    trim_amount = 0.0
    # this is a func that runs on HB, which needs trimming
    if root.func.startswith("@OFFLOAD_KERNEL@__") or root.func.startswith("@BSG_API_CALL@__"):
        assert len(root.children) == 1
        simulated = root.children[0]
        assert simulated.func == "@TRIM@"
        trim_amount = simulated.time - root.time
        root.time += trim_amount
        return trim_amount

    for kid in root.children:
        trim_amount += exec_time_apply_trim(kid)
    root.time += trim_amount
    return trim_amount

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
def exec_time_tree(raw_stack, fancy_func=False, trimming=False):
    data = raw_stack
    data, roi = exec_time_preprocess(data)
    root = exec_time_Node(roi[0], roi[1])
    exec_time_construct_tree_impl(data, root, fancy_func)
    if trimming:
        # simulation time trimming
        exec_time_apply_trim(root)
    exec_time_add_other(root)
    exec_time_calc_percentage(root)
    return root
