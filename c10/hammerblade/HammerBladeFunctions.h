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

/*
 * inlcude bsg_manycore.h here
 */
#include <bsg_manycore_cuda.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>

namespace c10 {
namespace hammerblade {

C10_HAMMERBLADE_API DeviceIndex device_count() noexcept;

C10_HAMMERBLADE_API DeviceIndex current_device();

C10_HAMMERBLADE_API void set_device(DeviceIndex device);

//----------------------------------------------------------------------------------------
// Interface to bsg_manycore runtime
//----------------------------------------------------------------------------------------

C10_HAMMERBLADE_API int memcpy_host_to_device(void *dst, const void *src, uint32_t nbytes);
C10_HAMMERBLADE_API int memcpy_device_to_host(void *dst, const void *src, uint32_t nbytes);

}} // namespace c10::hammerblade
