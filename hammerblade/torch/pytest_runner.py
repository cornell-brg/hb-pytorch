"""
In order to use pytest inside COSIM, we need a wrapper to invoke pytest for us

Feb 04, 2020
Lin Cheng
"""

# pytest commandline options
pytest_argv = ["-vs"]

# This is a work around of the bug in which sys.argv is not set
# when running as embededd script
import sys
if not hasattr(sys, "argv"):
    sys.argv = ["__main__.py"]

# import essential modules
import pathlib
import pytest

# figure out current directory
current_path = str(pathlib.Path(__file__).parent.absolute())

# add current and tests paths
sys.path.append(current_path + "/tests")
sys.path.append(current_path)

# construct target list
targets = []

# Get test list from the command line if provided
if len(sys.argv) > 1:
    for t in sys.argv[1:]:
        targets.append(current_path + "/tests/" + t)
else:
    # collect pytest files. somehow it can't do so automatically
    # inside COSIM
    import glob
    targets = glob.glob(current_path + "/tests/test_*.py")

print()
print(" files collected by pytest runner:")
print()
for t in targets:
    print(" " + t)
print()
print(" starting pytest ...")
print()

# invoke pytest main loop
#exit(pytest.main(pytest_argv + targets + ['-k torch_addmm_perf']))
exit(pytest.main(pytest_argv + targets + ['-k torch_addmm_perf']))
