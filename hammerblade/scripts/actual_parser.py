"""
This is a script for parsing input tensors (actuals)

06/05/2020 Lin Cheng (lc873@cornell.edu)
"""

class ATen_Tensor:
    def __init__(self, name, full, chunk):
        self.name = name
        self.full = full
        self.chunk = chunk

    def __str__(self):
        return "Actual {0} - full:{1} <> chunk:{2}".format(self.name, self.full, self.chunk)


# INPUT:  raw actual string of full and chunk sizes
# OUTPUT: a List of ATen_Tensor objects

def parse(full_actuals, chunk_actuals):
  # split into tensors -- since we have an extra <|> at the end, we need this [:-1]
    full_actuals = full_actuals.split("<|>")[:-1]
    chunk_actuals = chunk_actuals.split("<|>")[:-1]
    assert len(full_actuals) == len(chunk_actuals)
    actuals = []
    idx = 0
    while idx < len(full_actuals):
        f_name, f_size = full_actuals[idx].split(";")
        c_name, c_size = chunk_actuals[idx].split(";")
        assert f_name == c_name
        actuals.append(ATen_Tensor(f_name, f_size, c_size))

        # debug
        print(actuals[-1])

        idx += 1

    return actuals
