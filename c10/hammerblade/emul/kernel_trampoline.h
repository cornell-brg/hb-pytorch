//====================================================================
// kernel_trampoline
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
#include <bsg_manycore_errno.h>

extern std::map<std::string, std::function<int(uint32_t, uint64_t*)>> kernelMap;
extern std::vector<std::function<int(uint32_t, uint64_t*)>> enqueued_kernel;
extern std::vector<uint32_t>  enqueued_argc;
extern std::vector<uint64_t*> enqueued_argv;

void enqueue_kernel(const std::string &kernel, uint32_t argc, uint64_t* argv);
int execute_kernels();

typedef struct _kernel_registry_ {
    _kernel_registry_(std::string kernel_name, std::function<int(uint32_t, uint64_t*)> kernel_ptr) {
        kernelMap[kernel_name] = kernel_ptr;
    }
} kernel_registry;

#define HB_GET_MACRO(_1,_2,_3,_4,_5,NAME,...) NAME
#define HB_EMUL_REG_KERNEL(...) HB_GET_MACRO(__VA_ARGS__, HB_EMUL_REG_KERNEL_5ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_4ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_3ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_2ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_1ARGS)(__VA_ARGS__)      \

#define HB_EMUL_REG_KERNEL_1ARGS(kernel)                                                              \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 0);                                                                               \
    return kernel();                                                                                  \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \


#define HB_EMUL_REG_KERNEL_2ARGS(kernel, at0)                                                         \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 1);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    return kernel(arg0);                                                                              \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_3ARGS(kernel, at0, at1)                                                    \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 2);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    return kernel(arg0, arg1);                                                                        \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_4ARGS(kernel, at0, at1, at2)                                               \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 3);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    return kernel(arg0, arg1, arg2);                                                                  \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_5ARGS(kernel, at0, at1, at2, at3)                                          \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 4);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    return kernel(arg0, arg1, arg2, arg3);                                                            \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#endif // _KERNEL_JUMPSTARTER_H
