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
#include <string>
#include <bsg_manycore_errno.h>
#include <kernel_log.h>

extern std::map<std::string, std::function<int(uint32_t, uint64_t*)>> kernelMap;
extern std::vector<std::function<int(uint32_t, uint64_t*)>> enqueued_kernel;
extern std::vector<uint32_t>  enqueued_argc;
extern std::vector<uint64_t*> enqueued_argv;

// HB device kernel logger
#ifdef HB_ENABLE_KERNEL_LOG
  extern KernelLogger kernel_call_logger;
  #define LOG_KERNEL_CALL(...) kernel_call_logger.log_kernel_call(__VA_ARGS__)
#else // HB_ENABLE_KERNEL_LOG
  #define LOG_KERNEL_CALL(...)
#endif // HB_ENABLE_KERNEL_LOG

void enqueue_kernel(const std::string &kernel, uint32_t argc, uint64_t* argv);
int execute_kernels();

typedef struct _kernel_registry_ {
    _kernel_registry_(std::string kernel_name, std::function<int(uint32_t, uint64_t*)> kernel_ptr) {
        kernelMap[kernel_name] = kernel_ptr;
    }
} kernel_registry;

#define HB_GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,NAME,...) NAME
#define HB_EMUL_REG_KERNEL(...) HB_GET_MACRO(__VA_ARGS__, HB_EMUL_REG_KERNEL_12ARGS,                  \
                                                          HB_EMUL_REG_KERNEL_11ARGS,                  \
                                                          HB_EMUL_REG_KERNEL_10ARGS,                  \
                                                          HB_EMUL_REG_KERNEL_9ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_8ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_7ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_6ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_5ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_4ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_3ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_2ARGS,                   \
                                                          HB_EMUL_REG_KERNEL_1ARGS)(__VA_ARGS__)      \

#define HB_EMUL_REG_KERNEL_1ARGS(kernel)                                                              \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 0);                                                                               \
    int err = kernel();                                                                               \
    LOG_KERNEL_CALL(#kernel);                                                                          \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \


#define HB_EMUL_REG_KERNEL_2ARGS(kernel, at0)                                                         \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 1);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    int err = kernel(arg0);                                                                           \
    LOG_KERNEL_CALL(#kernel, arg0);                                                                   \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_3ARGS(kernel, at0, at1)                                                    \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 2);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    int err = kernel(arg0, arg1);                                                                     \
    LOG_KERNEL_CALL(#kernel, arg0, arg1);                                                             \
    return err;                                                                                       \
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
    int err = kernel(arg0, arg1, arg2);                                                               \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2);                                                       \
    return err;                                                                                       \
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
    int err = kernel(arg0, arg1, arg2, arg3);                                                         \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3);                                                 \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_6ARGS(kernel, at0, at1, at2, at3, at4)                                     \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 5);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    uint64_t _arg4 = argv[4];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    at4 arg4 = (at4)((intptr_t)_arg4);                                                                \
    int err = kernel(arg0, arg1, arg2, arg3, arg4);                                                   \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3, arg4);                                           \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_7ARGS(kernel, at0, at1, at2, at3, at4, at5)                                \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 6);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    uint64_t _arg4 = argv[4];                                                                         \
    uint64_t _arg5 = argv[5];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    at4 arg4 = (at4)((intptr_t)_arg4);                                                                \
    at5 arg5 = (at5)((intptr_t)_arg5);                                                                \
    int err = kernel(arg0, arg1, arg2, arg3, arg4, arg5);                                             \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3, arg4, arg5);                                     \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_8ARGS(kernel, at0, at1, at2, at3, at4, at5, at6)                           \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 7);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    uint64_t _arg4 = argv[4];                                                                         \
    uint64_t _arg5 = argv[5];                                                                         \
    uint64_t _arg6 = argv[6];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    at4 arg4 = (at4)((intptr_t)_arg4);                                                                \
    at5 arg5 = (at5)((intptr_t)_arg5);                                                                \
    at6 arg6 = (at6)((intptr_t)_arg6);                                                                \
    int err = kernel(arg0, arg1, arg2, arg3, arg4, arg5, arg6);                                       \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3, arg4, arg5, arg6);                               \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_9ARGS(kernel, at0, at1, at2, at3, at4, at5, at6, at7)                      \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 8);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    uint64_t _arg4 = argv[4];                                                                         \
    uint64_t _arg5 = argv[5];                                                                         \
    uint64_t _arg6 = argv[6];                                                                         \
    uint64_t _arg7 = argv[7];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    at4 arg4 = (at4)((intptr_t)_arg4);                                                                \
    at5 arg5 = (at5)((intptr_t)_arg5);                                                                \
    at6 arg6 = (at6)((intptr_t)_arg6);                                                                \
    at7 arg7 = (at7)((intptr_t)_arg7);                                                                \
    int err = kernel(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);                                 \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);                         \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_10ARGS(kernel, at0, at1, at2, at3, at4, at5, at6, at7, at8)                \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 9);                                                                               \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    uint64_t _arg4 = argv[4];                                                                         \
    uint64_t _arg5 = argv[5];                                                                         \
    uint64_t _arg6 = argv[6];                                                                         \
    uint64_t _arg7 = argv[7];                                                                         \
    uint64_t _arg8 = argv[8];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    at4 arg4 = (at4)((intptr_t)_arg4);                                                                \
    at5 arg5 = (at5)((intptr_t)_arg5);                                                                \
    at6 arg6 = (at6)((intptr_t)_arg6);                                                                \
    at7 arg7 = (at7)((intptr_t)_arg7);                                                                \
    at8 arg8 = (at8)((intptr_t)_arg8);                                                                \
    int err = kernel(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);                           \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);                   \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_11ARGS(kernel, at0, at1, at2, at3, at4, at5, at6, at7, at8, at9)           \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 10);                                                                              \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    uint64_t _arg4 = argv[4];                                                                         \
    uint64_t _arg5 = argv[5];                                                                         \
    uint64_t _arg6 = argv[6];                                                                         \
    uint64_t _arg7 = argv[7];                                                                         \
    uint64_t _arg8 = argv[8];                                                                         \
    uint64_t _arg9 = argv[9];                                                                         \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    at4 arg4 = (at4)((intptr_t)_arg4);                                                                \
    at5 arg5 = (at5)((intptr_t)_arg5);                                                                \
    at6 arg6 = (at6)((intptr_t)_arg6);                                                                \
    at7 arg7 = (at7)((intptr_t)_arg7);                                                                \
    at8 arg8 = (at8)((intptr_t)_arg8);                                                                \
    at9 arg9 = (at9)((intptr_t)_arg9);                                                                \
    int err = kernel(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);                     \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);             \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#define HB_EMUL_REG_KERNEL_12ARGS(kernel, at0, at1, at2, at3, at4, at5, at6, at7, at8, at9, at10)     \
int trampoline_##kernel(const uint32_t argc, const uint64_t* argv) {                                  \
    assert (argc == 11);                                                                              \
    uint64_t _arg0 = argv[0];                                                                         \
    uint64_t _arg1 = argv[1];                                                                         \
    uint64_t _arg2 = argv[2];                                                                         \
    uint64_t _arg3 = argv[3];                                                                         \
    uint64_t _arg4 = argv[4];                                                                         \
    uint64_t _arg5 = argv[5];                                                                         \
    uint64_t _arg6 = argv[6];                                                                         \
    uint64_t _arg7 = argv[7];                                                                         \
    uint64_t _arg8 = argv[8];                                                                         \
    uint64_t _arg9 = argv[9];                                                                         \
    uint64_t _arg10 = argv[10];                                                                       \
    at0 arg0 = (at0)((intptr_t)_arg0);                                                                \
    at1 arg1 = (at1)((intptr_t)_arg1);                                                                \
    at2 arg2 = (at2)((intptr_t)_arg2);                                                                \
    at3 arg3 = (at3)((intptr_t)_arg3);                                                                \
    at4 arg4 = (at4)((intptr_t)_arg4);                                                                \
    at5 arg5 = (at5)((intptr_t)_arg5);                                                                \
    at6 arg6 = (at6)((intptr_t)_arg6);                                                                \
    at7 arg7 = (at7)((intptr_t)_arg7);                                                                \
    at8 arg8 = (at8)((intptr_t)_arg8);                                                                \
    at9 arg9 = (at9)((intptr_t)_arg9);                                                                \
    at10 arg10 = (at10)((intptr_t)_arg10);                                                            \
    int err = kernel(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);              \
    LOG_KERNEL_CALL(#kernel, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);      \
    return err;                                                                                       \
}                                                                                                     \
kernel_registry registry_##kernel = {#kernel, trampoline_##kernel};                                   \

#endif // _KERNEL_JUMPSTARTER_H
