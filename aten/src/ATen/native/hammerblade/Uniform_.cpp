#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/HammerBladeGenerator.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor& uniform_(Tensor& self, float from, float to, Generator* gen) {

    AT_DISPATCH_FLOAT_TYPE_ONLY(self.scalar_type(), "uniform_", [&]() {
    HammerBladeGenerator* generator = get_generator_or_default<HammerBladeGenerator>(gen,
                 hammerblade::detail::getDefaultHammerBladeGenerator());

    //auto seed_tensor = hammerblade_common_seed_to_tensor(generator->random());


    hb_offload_kernel(self, from, to, "tensorlib_uniform_");
  });
  return self;
}
}} // namespace at::native

