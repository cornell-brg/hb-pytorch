#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor addmm_hb(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha
) {

  //TODO: implmement the correct addmm_hb host code
  //      you may use void offload_tensor_scalar_impl(std::vector<Tensor>, std::vector<Scalar>,
  //                                                  cosn char*)
  //      which is defined in Offload.h for kernel offloading
  return self;

}

}} // namespace at::native
