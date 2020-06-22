![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

## HB-PyTorch COSIM Tutorial
 - Author: Lin Cheng (lc873@cornell.edu)
 - Date: June 22, 2020
 
#### Table of Contents
 - Introduction
 - Setup COSIM (Bladerunner)
 - Build HB-Pytorch for cosimulation
 - Basic performance profiling
 - To Do On Your Own (optimize Vincr)
 
### Introduction
This tutorial will discuss how to setup cosimulation (bsg_bladerunner) environment, and build HB-Pytorch for cosimulation instead of emulation. Bladerunner is a RTL level cycle-accurate simulator for HammerBlade. It is called cosimulation because host code runs natively on CPU while HammerBalde device code runs on the simulator. We will also explore basic performance profiling and briefly introduce optimization methods.

### Setup COSIM (Bladerunner)
Follow these steps to setup a cosimulation environment.
- Clone bsg_bladerunner repo. If you work on brg-vip, clone the Cornell fork.

      git clone https://github.com/bespoke-silicon-group/bsg_bladerunner.git
      git clone git@github.com:cornell-brg/bsg_bladerunner.git (if you are using brg-vip)
      
- Make sure vivado and vcs are available on your machine

      which vivado
      which vcs

 - Build bsg_bladerunner. Most of the building time will be spent on compiling riscv-toolchain.
 
       cd bsg_bladerunner
       export TOP=$PWD
       export BRG_BSG_BLADERUNNER_DIR=$PWD
       make setup
 
 - Run regression tests to verify Bladerunner installation. This first time you do `make regression`, a number of Xilinx IPs need to be built. If you are using brg-vip or a machine from Prof. Zhang's group, it _will_ fail with a error in HDMI_controller. It is _safe_ to ignore this error. (Note: this file structure is being actively changed)
 
       cd $TOP/bsg_replicant/testbenches
       make regression
       make regression (do it again if the previous one failed due to errors in HDMI_controller)
       
If all tests are passed, you have successfully built bsg_bladerunner and ready to go.

### Build HB-Pytorch for cosimulation
Now we build HB-Pytorch to run on bsg_bladerunner.
- Create a [Python virtual environment][venv]:

      python3 -m venv ./venv_pytorch_cosim
      source ./venv_pytorch_cosim/bin/activate

- Install some dependencies:

      pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing sklearn tqdm pytest ninja hypothesis

- Clone this repository:

      git clone git@github.com:cornell-brg/hb-pytorch.git
      
- Change directory:

      cd hb-pytorch

- Init PyTorch third party dependencies:

      git submodule update --init --recursive

- Setup building environment variables. Before you run this script, make sure environment varibale BRG_BSG_BLADERUNNER_DIR points to your Bladerunner:

      source setup_cosim_build_env.sh
      
- Build PyTorch. This step can take up to 15 minutes and it is usually longer than building for emulation:

      python setup.py develop
      
- Varify PyTorch with HB support is correctly built -- HB-Pytorch should work correctly only if is run with COSIM.

      cd ../
      python
      Python 3.7.4 (default, Sep 28 2019, 14:22:12)
      [GCC 6.3.1 20170216 (Red Hat 6.3.1-3)] on linux
      >>> import torch
      >>> torch.hammerblade.init()
      python: symbol lookup error: /work/global/brg/install/bare-pkgs/x86_64-centos7/bladerunner/bsg_replicant/libraries/libbsg_manycore_runtime.so.1: undefined symbol: svGetScopeFromName
      
- Run a test script on Bladerunner. We use `pycosim` instead of `python` to invoke a script.
      
      echo "import torch; torch.hammerblade.init(); print(torch.__config__.show());" > /tmp/test.py
      pycosim /tmp/test.py

If everything is correct, you should see
```
PyTorch configed with 4 * 4 HB device
HB startup config kernel applied
PyTorch built with:
  - GCC 8.3
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - Build settings: BLAS=MKL, BUILD_NAMEDTENSOR=OFF, BUILD_TYPE=Debug, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -fopenmp -DUSE_PYTORCH_QNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Wno-stringop-overflow, DISABLE_NUMA=1, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=0, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=0, USE_OPENMP=ON, USE_STATIC_DISPATCH=OFF,
```

Note that the print out says we have config'ed a 4x4 HB device. Where do we say that? To choose which machine to simulate, edit

      vim $TOP/bsg_replicant/machine.mk

Available machines are located in `$TOP/bsg_replicant/machines`. To test your kernel, we usually use `4x4_fast_n_fake`.

[venv]: https://docs.python.org/3/tutorial/venv.html

### Basic performance profiling
Now we are ready to use COSIM for performance profiling. We will take our `Vincr` as an example. We first write a small script to exercise it.

      import torch
      x = torch.ones(1000).hammerblade
      torch.vincr(x)
      
Save the file as `test_vincr.py`. As we have mentioned before, we run the script with `pycosim`

      pycosim test_vincr.py
      
After it finishes running, you should find a number of files under the path you invoked `pycosim`

      ls ./
      ucli.key
      vanilla.log
      vanilla_stats.csv
      vcache_operation_trace.csv
      vcache_stats.csv
      ...
      
We mostly care about `vanilla_stats.csv`. Take a look inside if you want, but it is not very human readable. We use UW's data parser to generate a nicely looking log for us

      python $TOP/bsg_manycore/software/py/vanilla_parser/stats_parser.py --stats vanilla_stats.csv
      
You should find a new folder named `stats`. Open the log and take a look

      vim stats/manycore_stats.log
      
Here is an example
```
Per-Tag Stats
Tag ID                 Aggr Instructions      Aggr I$ Misses   Aggr Stall Cycles  Aggr Bubble Cycles   Aggr Total Cycles    Abs Total Cycles                 IPC    % of Kernel Cycles
======================================================================================================================================================================================
kernel                           2414706               13301            21279062               55260            23762331              421937              0.1016              100.00
======================================================================================================================================================================================

Per-Tile-Group Timing Stats
Tile Group ID                         Aggr Instructions   Aggr Total Cycles     Abs Total Cycle                 IPC   TG / Tag-Total (%)   TG / Kernel-Total(%)
======================================================================================================================================================================================
-----------------------------------------------------------------------------------  Tag kernel    -----------------------------------------------------------------------------------
0                                               2414706            23762331              421937              0.1016              100.00              100.00
total                                           2414706            23762331              421937              0.1016              100.00              100.00
======================================================================================================================================================================================

Per-Tag Miss Stats
Miss Type                                        Misses            Accesses        Hit Rate (%)
======================================================================================================================================================================================
-----------------------------------------------------------------------------------  Tag kernel    -----------------------------------------------------------------------------------
miss_icache                                       13301             2414706               99.45
miss_beq                                            129                 390               66.92
miss_bne                                            387              267726               99.86
miss_blt                                              3                   6               50.00
miss_bge                                              2                   4               50.00
miss_bltu                                           128                 384               66.67
miss_bgeu                                           383                 767               50.07
miss_jalr                                             0                   1              100.00
miss_total                                        14333              282579               94.93
======================================================================================================================================================================================

Per-Tag Stall Stats
Stall Type                                       Cycles      % Stall Cycles      % Total Cycles
======================================================================================================================================================================================
-----------------------------------------------------------------------------------  Tag kernel    -----------------------------------------------------------------------------------
stall_fp_remote_load                                  0                0.00                0.00
stall_fp_local_load                                   0                0.00                0.00
stall_depend                                   20896383               98.20               87.94
  stall_depend_remote_load_dram                19829091               93.19               83.45
  stall_depend_remote_load_global                     0                0.00                0.00
  stall_depend_remote_load_group                      0                0.00                0.00
  stall_depend_local_load                        266823                1.25                1.12
stall_force_wb                                        0                0.00                0.00
stall_ifetch_wait                                358999                1.69                1.51
stall_icache_store                                    0                0.00                0.00
stall_lr_aq                                           0                0.00                0.00
stall_md                                          23680                0.11                0.10
stall_remote_req                                      0                0.00                0.00
stall_local_flw                                       0                0.00                0.00
stall_amo_aq                                          0                0.00                0.00
stall_amo_rl                                          0                0.00                0.00
stall_total                                    21279062              100.00               89.55
not_stall                                       2483269               11.67               10.45
======================================================================================================================================================================================

Per-Tag Bubble Stats
Bubble Type                                      Cycles        % of Bubbles   % of Total Cycles
======================================================================================================================================================================================
-----------------------------------------------------------------------------------  Tag kernel    -----------------------------------------------------------------------------------
bubble_icache                                     53196               96.26                0.22
bubble_branch_mispredict                           2064                3.74                0.01
bubble_jalr_mispredict                                0                0.00                0.00
bubble_total                                      55260              100.00                0.23
======================================================================================================================================================================================

Per-Tag Instruction Stats
Instruction                                       Count   % of Instructions
======================================================================================================================================================================================
-----------------------------------------------------------------------------------  Tag kernel    -----------------------------------------------------------------------------------
instr_fadd                                            0                0.00
instr_fsub                                            0                0.00
instr_fmul                                       266951               11.06
instr_fsgnj                                           0                0.00
instr_fsgnjn                                          0                0.00
instr_fsgnjx                                          0                0.00
instr_fmin                                            0                0.00
instr_fmax                                            0                0.00
instr_fcvt_s_w                                        0                0.00
instr_fcvt_s_wu                                       0                0.00
instr_fmv_w_x                                         1                0.00
instr_feq                                             0                0.00
instr_flt                                             0                0.00
instr_fle                                             0                0.00
instr_fcvt_w_s                                        0                0.00
instr_fcvt_wu_s                                       0                0.00
instr_fclass                                          0                0.00
instr_fmv_x_w                                         0                0.00
instr_local_ld                                     1954                0.08
instr_local_st                                      518                0.02
instr_remote_ld_dram                                386                0.02
instr_remote_ld_global                                0                0.00
instr_remote_ld_group                                 0                0.00
instr_remote_st_dram                                  0                0.00
instr_remote_st_global                              256                0.01
instr_remote_st_group                                 0                0.00
instr_local_flw                                  266951               11.06
instr_local_fsw                                     128                0.01
instr_remote_flw                                 267079               11.06
instr_remote_fsw                                 266952               11.06
instr_lr                                              0                0.00
instr_lr_aq                                           0                0.00
  instr_amoswap                                       0                0.00
  instr_amoor                                         0                0.00
instr_beq                                           261                0.01
instr_bne                                        267339               11.07
instr_blt                                             3                0.00
instr_bge                                             2                0.00
instr_bltu                                          256                0.01
instr_bgeu                                          384                0.02
instr_jalr                                            1                0.00
instr_jal                                           259                0.01
instr_sll                                             0                0.00
instr_slli                                          781                0.03
instr_srl                                             0                0.00
instr_srli                                          262                0.01
instr_sra                                             0                0.00
instr_srai                                            0                0.00
instr_add                                        802009               33.21
instr_addi                                       268237               11.11
instr_sub                                             2                0.00
instr_lui                                          1544                0.06
instr_auipc                                           0                0.00
instr_xor                                             0                0.00
instr_xori                                            0                0.00
instr_or                                            776                0.03
instr_ori                                             1                0.00
instr_and                                           769                0.03
instr_andi                                            3                0.00
instr_slt                                             0                0.00
instr_slti                                            0                0.00
instr_sltu                                            1                0.00
instr_sltiu                                           0                0.00
instr_mul                                           640                0.03
instr_mulh                                            0                0.00
instr_mulhsu                                          0                0.00
instr_mulhu                                           0                0.00
instr_div                                             0                0.00
instr_divu                                            0                0.00
instr_rem                                             0                0.00
instr_remu                                            0                0.00
instr_fence                                           0                0.00
instr_total                                     2414706              100.00
======================================================================================================================================================================================
```

### To Do On Your Own (optimize Vincr)
Now it's time for you to do something -- let's parallelize our `Vincr` kernel! After you have done so, compare the performance of both version by looking at `Aggr Instructions` and `Abs Total Cycles` fields. Good luck!
