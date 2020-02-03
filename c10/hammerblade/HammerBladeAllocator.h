#pragma once

#include <cstring>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>
#include <c10/util/numa.h>

namespace c10 {

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
C10_API void NoDelete(void*);

C10_API void* alloc_hb(size_t nbytes);
C10_API void free_hb(void* data);

// Get the HammerBlade Allocator
C10_API at::Allocator* GetHammerBladeAllocator();

} // namespace c10
