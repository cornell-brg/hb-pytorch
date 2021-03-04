import torch

class exec_time_Node:

    def __init__(self, func, time):
        self.func = func
        self.time = float(time)
        self.children = []
        self.percentage = -1

    def add_child(self, child):
        assert isinstance(child, exec_time_Node)
        self.children.append(child)

    def __str__(self):
        return "Node(" + self.func + " : " + str(self.time) + ")"


def exec_time_construct_tree_impl(data):
  data = data.splitlines()
  ROI_data = data[0].split(";")
  assert ROI_data[0] == "ROI"
  ROI = exec_time_Node("ROI",ROI_data[1])

  # helper
  def node_adder(path, parent, time):
    # base
    if len(path) == 1:
      # the node we want to add shouldn't already there
      for kid in parent.children:
        assert kid.func != path[0]
      parent.add_child(exec_time_Node(path[0],time))
      return
    # find the kid in parent's children list
    for kid in parent.children:
      if kid.func == path[0]:
        node_adder(path[1:], kid, time)
        return
    # not found ...
    assert False

  # process each entry
  for d in data[1:]:
    d = d.split(";")
    path = d[0]
    time = d[1]
    path = path.split("<|>")
    node_adder(path[1:], ROI, time)

  return ROI

# build total time from ground up
# a simple postorder traversla will do
def accumulate_time(root):
  children_time = 0
  for kid in root.children:
    children_time += accumulate_time(kid)
  root.time += children_time
  return root.time

# find other time in ROI
def exec_time_add_other(root):
    agg_total = 0.0
    for kid in root.children:
        agg_total += kid.time
    root.children.append(exec_time_Node("other", root.time - agg_total))

# deal with TRIM nodes
# if trimming we adjust the parent
# if not we set trim value to 0
def adjust_trimming(root):
  for kid in root.children:
    if kid.func == "@TRIM@":
      assert len(root.children) == 1 # the only kid should be trim, if there is a trim
      root.time = kid.time # adjust time to simulated time
      kid.time = 0 # disgrad this time to prevent double counting
    else:
      adjust_trimming(kid)

def disgrad_trimming(root):
  # base
  if root.func == "@TRIM@":
    root.time = 0 # we disgrad this info
    assert len(root.children) == 0
    return
  # recursion
  for kid in root.children:
    disgrad_trimming(kid)

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
def exec_time_tree(trimming=False):
    data = torch._C._hb_profiler_exec_time_raw_stack()
    print(data)
    root = exec_time_construct_tree_impl(data)
    if trimming:
      adjust_trimming(root)
    else:
      disgrad_trimming(root)
    accumulate_time(root)
    exec_time_add_other(root)
    exec_time_calc_percentage(root)
    return root

# --------- torch.hb_profiler.exec_time APIs ---------

def fancy_print(trimming=False):
    try:
        root = exec_time_tree(trimming=trimming)
        buffer = ""
        for e in (root.children + [root]):
            func = e.func
            time = e.time / 1000000.0
            percentage = e.percentage
            buffer += ('{func:30}     {time:.2f} {percentage:.1f}%\n'.format(
                func=func, time=time, percentage=percentage))
        return buffer
    except AttributeError:
        print("PyTorch is not built with profiling")

def latex_table(trimming=False):
    try:
        root = exec_time_tree(trimming=trimming)
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
            time = e.time / 1000000.0
            percentage = e.percentage
            buffer += ('\\textbf{{{func:30}}} &  {time:.2f} & {percentage:.1f}\\% \\\\\n'.format(
                func=func, time=time, percentage=percentage))

        footer = "\\bottomrule\n" \
                 "\\end{tabular}\n" \
                 "\\label{tbl-plat}\n" \
                 "\\end{table}\n"
        buffer += footer
        return buffer
    except AttributeError:
        print("PyTorch is not built with profiling")

def raw_stack(trimming=False):
    try:
        root = exec_time_tree(trimming=trimming)
        return exec_time_print_tree(root)
    except AttributeError:
        print("PyTorch is not built with profiling")

def roi_time(trimming=False):
    try:
        root = exec_time_tree(trimming=trimming)
        if trimming:
            exec_time_apply_trim(root)
        return root.time
    except AttributeError:
        print("PyTorch is not built with profiling")

def exec_time_dict(trimming=False):
    """
    Returns a dict with per operator execution times for all
    operators on the top of the stack.
    """
    try:
        root = exec_time_tree(fancy_func=True, trimming=trimming)
        exec_dict = {}
        for e in (root.children + [root]):
            exec_dict[e.func] = e.time
        return exec_dict
    except AttributeError:
        print("PyTorch is not built with profiling")
