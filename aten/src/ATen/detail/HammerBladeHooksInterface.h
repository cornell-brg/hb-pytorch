#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

// Forward-declares THBState
struct THBState;

namespace at {
class Context;
}

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

// HammerBladeHooksInterface is an omnibus interface for any HB functionality
// withc we may want to call into from CPU code. This is the same as PyTorch's
// CUDA backend.
// TODO: currently we are not compiling HB library as a standalone shared
// library. I injected HB related code into CPU library. In the future, we
// may want to get it out.
struct CAFFE2_API HammerBladeHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~HammerBladeHooksInterface() {}

  // Initialize THBState and, transitively, the HB state
  virtual std::unique_ptr<THBState, void (*)(THBState*)> initHammerBlade() const {
    TORCH_CHECK(false, "Cannot initialize HammerBlade without ATen_hammerblade library.");
  }

  virtual std::unique_ptr<Generator> initHammerBladeGenerator(Context*) const {
    TORCH_CHECK(false, "Cannot initialize HammerBlade generator without ATen_hammerblade library.");
  }

  virtual Generator* getDefaultHammerBladeGenerator(DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default HammerBlade generator without ATen_hammerblade library.");
  }

  virtual Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK(false, "Cannot get device of pointer on HammerBlade without ATen_hammerblade library.");
  }

  virtual bool isPinnedPtr(void* data) const {
    return false;
  }

  virtual bool hasHammerBlade() const {
    return false;
  }

  virtual int64_t current_device() const {
    return -1;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "HammerBlade does not support pinned memory allocator");
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed HammerBlade version without ATen_hammerblade library.");
  }

  virtual int getNumHBDevices() const {
    return 0;
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct CAFFE2_API HammerBladeHooksArgs {};

C10_DECLARE_REGISTRY(HammerBladeHooksRegistry, HammerBladeHooksInterface, HammerBladeHooksArgs);
#define REGISTER_HAMMERBLADE_HOOKS(clsname) \
  C10_REGISTER_CLASS(HammerBladeHooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const HammerBladeHooksInterface& getHammerBladeHooks();
} // namespace detail
} // namespace at
