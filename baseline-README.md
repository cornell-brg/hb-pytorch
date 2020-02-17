# Baseline Tests for HammerBlade Manycore

This repository contains performance tests for the HammerBlade
Manycore Architecture. It is intended to be used as a submodule within
the BSG Bladerunner Architecture, which contains pointers to
known-good commits in BaseJump STL, BSG Manycore, and BSG Replicant. 

## Contents

This repository contains the following folders: 

- `examples`: Example software for the HammerBlade Manycore, written
  using the CUDA-Lite Runtime. CUDA-Lite Kernel Sources and Host
  CUDA-Lite Sources are co-located in each sub-directory.

- `fragments`: Makefile fragments that support the programs in this
  repository. The fragments can build Manycore Binaries from CUDA-Lite
  Sources and Host executables for launching programs.

This repository contains the following files:

- `README.md`: This file
- `environment.mk`: A makefile fragment for deducing the build environment. 
- `hdk.mk`: A makefile fragment for deducing the AWS-FPGA HDK build environment.

## Dependencies

This repository should be cloned inside of [BSG
Bladerunner](https://github.com/bespoke-silicon-group/bsg_bladerunner).

To simulate/co-simulate/build these projects you must have the
following tools. They are also listed in the BSG Bladerunner
repository.

   1. Vivado 2019.1 (With the following [bug-fix](https://www.xilinx.com/support/answers/72404.html) applied)
   3. Synopsys VCS (We use O-2018.09-SP2, but others would work)

## Setup

1. Setup the [BSG Bladerunner](https://github.com/bespoke-silicon-group/bsg_bladerunner) repository.

2. Follow the [Setup](https://github.com/bespoke-silicon-group/bsg_bladerunner/#setup) instructions in BSG Bladerunner

3. Run `make default` from inside of one of the applications inside of the `examples` directory.

## Post-Script

Baseline is a reference to the film Bladerunner 2049. 

*The Baseline is designed to test the effects of a replicant's job on
their brain and psyche, because of the nature of their job, they
constantly need to be assessed as to whether their work is having some
kind of moral impact on them.*

In a much less macabre way, these test programs should be used to
measure performance of the HammerBlade Manycore as the architecture
develops.