![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

# PyTorch HammerBlade Port
This work aims to port PyTorch to HammerBlade.

### How to build
  This assumes that you have a working COSIM installed. Then you can either put `hb-pytorch` under `bsg_bladerunner`, or set `BRG_BSG_BLADERUNNER_DIR` to your `bladerunner` path.
 - Clone hb-pytorch repo
    `git clone -b hb-device git@github.com:cornell-brg/hb-pytorch.git`
 - Create python virtual environment
    `python3.6 -m venv ./venv_pytorch`
    `python3.6 -m venv ./venv_pytorch`
 - Install dependencies
    `pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing sklearn tqdm pytest`
 - Init pytorch third party dependencies
    `git submodule update --init --recursive`
 - Setup building environment variables. You need to edit `hb-pytorch/setup_cosim_build_env.sh` and set `BSG_MANYCORE_DIR` to `<bsg_bladerunner>/bsg_replicant/libraries`
    `cd hb-pytorch && source setup_cosim_build_env.sh`
 - Build pytorch. This step can take up to 15 minutes
    `python setup.py install`

### Run Pytests
 - Goto hb-pytorch directory
    `cd hb-pytorch`
 - Run pytest
    `source ./run-hb-pytest.sh`


### Important files and directories related to HammerBlade
#### files used to run pytest (adapted from Baseline)
  - `hammerblade/fragments/`
  - `hammerblade/environment.mk`
  - `baseline-README.md`
  - `run-hb-pytest.sh` (`source` this one to run pytest!)
  - `hammerblade/torch/`
#### HammerBlade device code
  - `hammerblade/torch/kernel`
#### Pytest tests
  - `hammerblade/torch/tests/`
  - `hammerblade/torch/tests/targets.py` (Register a new test here!)
#### files that interacts with HammerBlade CUDALite runtime
  - `c10/hammerblade/`
