#pragma once

// This header provides C++ wrappers around commonly used HB API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

// #include <c10/macros/Macros.h>
#include <c10/hammerblade/HammerBladeMacros.h>
#include <c10/hammerblade/HammerBladeDevice.h>
#include <c10/hammerblade/HammerBladeException.h>
#include <c10/core/Device.h>
#include <atomic>

/*
 * inlcude bsg_manycore.h here
 */
#include <bsg_manycore_cuda.h>
#include <bsg_manycore_printing.h>

namespace c10 {
namespace hammerblade {

C10_HAMMERBLADE_API DeviceIndex device_count() noexcept;

C10_HAMMERBLADE_API DeviceIndex current_device();

C10_HAMMERBLADE_API void set_device(DeviceIndex device);

//----------------------------------------------------------------------------------------
// Interface to bsg_manycore runtime
//----------------------------------------------------------------------------------------

C10_HAMMERBLADE_API eva_t device_malloc(size_t nbytes);
C10_HAMMERBLADE_API void device_free(eva_t data);
C10_HAMMERBLADE_API void* memcpy_host_to_device(void *dst, const void *src, uint32_t nbytes);
C10_HAMMERBLADE_API void* memcpy_device_to_host(void *dst, const void *src, uint32_t nbytes);
C10_HAMMERBLADE_API void* DMA_host_to_device(void *dst, const void *src, uint32_t nbytes);
C10_HAMMERBLADE_API void* DMA_device_to_host(void *dst, const void *src, uint32_t nbytes);
C10_HAMMERBLADE_API void offload_kernel(const char* kernel, std::vector<eva_t> args);

extern std::atomic<int> hb_device_status;

}} // namespace c10::hammerblade
