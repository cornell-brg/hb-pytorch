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

#define PATH(x) pstr(x)
#define pstr(x) #x
static char kernel_json[] = PATH(HB_KERNEL_LOG);
#undef pstr
#undef PATH

void kernel_log(hb_tensor_t* arg);
void kernel_log(float* arg);

#endif // _KERNEL_LOG_H
