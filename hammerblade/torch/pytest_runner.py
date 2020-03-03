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

# figure out regression/pytorch directory
regression_path = str(pathlib.Path(__file__).parent.absolute())

# use regression/pytorch so we can find out tests
sys.path.append(regression_path + "/tests")
sys.path.append(regression_path)

# load registered tests
from tests.targets import pytest_targets

# construct target list
targets = []
for t in pytest_targets:
    targets.append(regression_path + "/tests/" + t + ".py")

# invoke pytest main loop
pytest.main(pytest_argv + targets)
