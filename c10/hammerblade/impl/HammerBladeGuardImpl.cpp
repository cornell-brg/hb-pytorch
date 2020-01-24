#include <c10/hammerblade/impl/HammerBladeGuardImpl.h>

namespace c10 {
namespace hammerblade {
namespace impl {

  constexpr DeviceType HAMMERBLADEGuardImpl::static_type;

  C10_REGISTER_GUARD_IMPL(HAMMERBLADE, HAMMERBLADEGuardImpl);

} // namespace impl
} // namespace hammerblade
} // namespace c10
