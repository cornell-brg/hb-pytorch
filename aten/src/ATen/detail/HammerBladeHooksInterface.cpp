#include <ATen/detail/HammerBladeHooksInterface.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <memory>
#include <mutex>

namespace at {
namespace detail {

static HammerBladeHooksInterface* hammerblade_hooks = nullptr;

const HammerBladeHooksInterface& getHammerBladeHooks() {
  static std::once_flag once;
  std::call_once(once, [] {
    hammerblade_hooks = HammerBladeHooksRegistry()->Create("HammerBladeHooks", HammerBladeHooksArgs{}).release();
    if (!hammerblade_hooks) {
      hammerblade_hooks = new HammerBladeHooksInterface();
    }
  });
  return *hammerblade_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(HammerBladeHooksRegistry, HammerBladeHooksInterface, HammerBladeHooksArgs)

} // namespace at
