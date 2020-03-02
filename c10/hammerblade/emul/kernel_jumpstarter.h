//====================================================================
// kernel_jumpstarter
// 03/02/2020, Lin Cheng (lc873@cornell.edu)
//====================================================================
// This file hosts the functions and data structures that are essential
// to enqueue and tile_execute simulations
//
// When a kernel is enqueued, we push its argc, argv, and its function
// pointer to corresponding queues.
//
// When tile_execute is called, we execute all enqueued functions in
// order.

#ifndef _KERNEL_JUMPSTARTER_H
#define _KERNEL_JUMPSTARTER_H

#include <map>
#include <vector>
#include <cassert>
#include <functional>
#include <kernel.h>
#include <bsg_manycore_errno.h>

extern std::map<std::string, std::function<int(uint32_t, uint32_t*)>> kernelMap;
extern std::vector<std::function<int(uint32_t, uint32_t*)>> enqueued_kernel;
extern std::vector<uint32_t>  enqueued_argc;
extern std::vector<uint32_t*> enqueued_argv;

void enqueue_kernel(const std::string &kernel, uint32_t argc, uint32_t* argv);
int execute_kernels();

#endif // _KERNEL_JUMPSTARTER_H
