#pragma once

// This header provides C++ wrappers around commonly used HB API functions.
// The benefit of using C++ here is that we can raise an exception in the
// event of an error, rather than explicitly pass around error codes.  This
// leads to more natural APIs.
//
// The naming convention used here matches the naming convention of torch.cuda

/*
 * inlcude bsg_manycore.h here
 * #include <bsg_manycore.h>
 */

// #include <c10/macros/Macros.h>
#include <c10/hammerblade/HammerBladeMacros.h>
#include <c10/core/Device.h>

namespace c10 {
namespace hammerblade {

C10_HAMMERBLADE_API DeviceIndex device_count() noexcept;

C10_HAMMERBLADE_API DeviceIndex current_device();

C10_HAMMERBLADE_API void set_device(DeviceIndex device);

}} // namespace c10::hammerblade
