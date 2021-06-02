"""
This is a script for extracting what we care about from out.std

06/05/2020 Lin Cheng (lc873@cornell.edu)
"""

# INPUT:  raw text of out.std
# OUTPUT: string of actuals and stack

def parse(out_std):
    # return values -- actuals & stack
    actuals = None
    stack_start = None
    stack_end = None
    stack = None
    # split input out.std logs into lines
    data = out_std.splitlines()
    idx = 0
    for d in data:
        print(d)
        if d.startswith("@#ACTUALS#@__"):
            # assuming exactly one redispatching per out.std log
            assert actuals is None
            actuals = d.split("@#ACTUALS#@__")[1]
        if d.startswith("#TOP_LEVEL_FUNC#__"):
            assert stack_start is None
            stack_start = idx
        if d.startswith("#TOP_LEVEL_FUNC_END#__"):
            assert stack_end is None
            stack_end = idx
        idx += 1
    # correctness check -- all of these should be true
    assert actuals is not None
    assert stack_start is not None
    assert stack_end is not None
    assert stack_start >=0
    assert stack_start < len(data)
    assert stack_end >=0
    assert stack_end < len(data)
    assert stack_end > stack_start
    stack = "\n".join(data[stack_start:stack_end + 1])

    return actuals, stack
