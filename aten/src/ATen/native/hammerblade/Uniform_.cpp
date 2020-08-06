#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/HammerBladeGenerator.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/Distributions.h>

namespace at { namespace native {

Tensor hammerblade_common_seed_to_tensor_(uint32_t seed = 42) {
  uint32_t upper = (uint32_t)(seed << 16);
  auto tensor = at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kInt));
  c10::hammerblade::memcpy_host_to_device(tensor.data_ptr(), (void*)&upper, sizeof(uint32_t));
  return tensor;
}

Tensor& uniform_(Tensor& self, double from, double to, Generator* gen) {

  AT_DISPATCH_FLOAT_TYPE_ONLY(self.scalar_type(), "uniform_", [&]() {
  HammerBladeGenerator* generator = get_generator_or_default<HammerBladeGenerator>(gen,
                 hammerblade::detail::getDefaultHammerBladeGenerator());

  float from_float = safe_downcast<float, double>(from);
  float to_float = safe_downcast<float, double>(to);

  auto seed_tensor = hammerblade_common_seed_to_tensor_(generator->random());

  hb_offload_kernel(self, seed_tensor, from_float, to_float, "tensorlib_uniform_");
  });
  return self;
}
}} // namespace at::native

