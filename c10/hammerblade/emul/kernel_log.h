//====================================================================
// Kernel Logger
//
// Functions and data structure to log kernel calls in emulation.
//
// 04/24/2020 Bandhav Veluri
//====================================================================

#ifndef _KERNEL_LOG_H_
#define _KERNEL_LOG_H_

#include <kernel_common.hpp>

void kernel_log(const char*);
void kernel_log(hb_tensor_t* arg);
void kernel_log(hb_vector_t* arg);
void kernel_log(float* arg);
void kernel_log(int32_t* arg);
void kernel_log(uint32_t* arg);

#endif // _KERNEL_LOG_H
