#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/HammerBladeGenerator.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

// ======================================================
// Common random seed generator.
//
// Input:  uint32_t random number by global RNG
// Output: 0D Tensor which stores the shifted common random seed
// | < - 16bit -> | < - 16bit -> |
//        RNG          bsg_id
// ======================================================

Tensor hammerblade_common_seed_to_tensor(uint32_t seed = 42) {
  uint32_t upper = (uint32_t)(seed << 16);
  auto tensor = at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kInt));
  c10::hammerblade::memcpy_host_to_device(tensor.data_ptr(), (void*)&upper, sizeof(uint32_t));
  return tensor;
}

// ======================================================
// The idea here is to use a generator on the host to generate
// a common seed for each HammerBlade tile (see above)
//
// Then each tile will add its tile id to this common seed
// and intialize its tile-local random number generator.
//
// We communicate this common seed to HammerBlade by wrapping
// it into a Tensor
// ======================================================

Tensor& bernoulli_scalar_hb_(Tensor& self, double p, Generator* gen) {
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  AT_DISPATCH_FLOAT_TYPE_ONLY(self.scalar_type(), "bernoulli_scalar_hb_", [&]() {
    HammerBladeGenerator* generator = get_generator_or_default<HammerBladeGenerator>(gen,
                                      hammerblade::detail::getDefaultHammerBladeGenerator());
    auto p_scalar = Scalar(p);
    auto seed_tensor = hammerblade_common_seed_to_tensor(generator->random());

    hb_offload_kernel(self, seed_tensor, p_scalar, "tensorlib_bernoulli_scalar_");
  });
  return self;
}

}} // namespace at::native
