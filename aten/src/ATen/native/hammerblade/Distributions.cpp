#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/HammerBladeGenerator.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/hammerblade/Offload.h>

#include <iostream>

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
  uint32_t* data = (uint32_t*)(tensor.data_ptr());
  *data = upper;
  return tensor;
}



Tensor& bernoulli_scalar_hb_(Tensor& self, double p, Generator* gen) {
  TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  AT_DISPATCH_FLOAT_TYPE_ONLY(self.scalar_type(), "bernoulli_scalar_hb_", [&]() {
    HammerBladeGenerator* generator = get_generator_or_default<HammerBladeGenerator>(gen,
                                      hammerblade::detail::getDefaultHammerBladeGenerator());
    auto p_scalar = Scalar(p);
    //auto common_seed = generator->next_wrapped_seed(); //at::empty({}, at::TensorOptions(at::kHAMMERBLADE).dtype(at::kInt));
    auto seed_tensor = hammerblade_common_seed_to_tensor(generator->random());
    // do arguments manually
    std::vector<Tensor> tensors;
    tensors.push_back(self);
    tensors.push_back(seed_tensor);
    std::vector<Scalar> scalars;
    scalars.push_back(p_scalar);

    std::cout << "scalar type = " << self.scalar_type() << std::endl;

    offload_tensor_scalar_impl(tensors, scalars, "tensorlib_bernoulli_scalar_");
  });
  return self;
}

}} // namespace at::native
