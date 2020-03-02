//====================================================================
// kernel.h
// 03/02/2020, Lin Cheng (lc873@cornell.edu)
//====================================================================
// In order to do kernel emulation, we use this init_kernel_starters
// function to populate the kernel function name to function pointer
// mapping.
//
// Note: the kernel function author is respondable for writting the
// mapping.
//
// init_kernel_starters is called when emulating hb_mc_device_program_init
// in bsg_manycore_cuda.h/cpp

#ifndef _KERNEL_TEST_H
#define _KERNEL_TEST_H

#include <kernel_jumpstarter.h>

void init_kernel_starters();

#endif
