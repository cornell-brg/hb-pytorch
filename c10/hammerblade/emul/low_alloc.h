//====================================================================
// low_alloc
// 03/03/2020, Lin Cheng (lc873@cornell.edu)
//====================================================================
// The idea is to request a memory region in low range using mmap
// so we can be sure that memory allocated on "device" can fit into
// uint32_t
//
// Note: this is a dummy linear allocation allocator, which means
// low_free does nothing

#ifndef _LOW_ALLOC_H
#define _LOW_ALLOC_H

#include <cstdlib>

void create_low_buffer();
void* low_malloc(size_t nbytes);
void low_free(void* addr);

#endif
