#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/ExpandUtils.h>

namespace at { namespace native {

Tensor addmv_hb(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    Scalar beta,
    Scalar alpha
) {

  if ( (self.scalar_type() != ScalarType::Float)
    || (mat.scalar_type() != ScalarType::Float)
    || (vec.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade addmv is implemented for Float only");
  }

  using scalar_t = float;

  TORCH_CHECK(mat.dim() == 2 && vec.dim() == 1, "2D matrix and 1D vector expected, got ", mat.dim(), " and ", vec.dim(), " tensors");

  TORCH_CHECK(mat.size(1) == vec.size(0), "Argument #3: Expected dim 0 size ", mat.size(1), ", got ", vec.size(0));

  Tensor b_self;
  std::tie(b_self) = expand_size(self, {mat.size(0)}, "addmv_out");
  Tensor result = at::empty({b_self.size(0)}, b_self.options());

  // Temporary tensor to store accumulated values : total rows x bsg_Y
  Tensor acc = at::empty({mat.size(0), 8}, b_self.options());

  /*
  //-------------------------------------
  // special cases
  // beta = 0, alpha = 1 -> same as mm
  // beta = 1, alpha = 1 -> naive_addmm
  //-------------------------------------
  if (beta.to<scalar_t>() == 0.0f && alpha.to<scalar_t>() == 1.0f) {
    // beta = 0, alpha = 1
    hb_offload_kernel(result, mat1, mat2, "tensorlib_mm");
  } else if (beta.to<scalar_t>() == 1.0f && alpha.to<scalar_t>() == 1.0f) {
    // beta = 1, alpha = 1
    hb_offload_kernel(result, b_self, mat1, mat2, "tensorlib_addmm_naive");
  } else {
    // general case
    hb_offload_kernel(result, b_self, mat1, mat2, beta.to<scalar_t>(),
                      alpha.to<scalar_t>(), "tensorlib_addmm");
  }
  */

  // Data parallel version of addmv for size of matrix > size of machine
  hb_offload_kernel(result, b_self, mat, vec, acc, beta.to<scalar_t>(),
    alpha.to<scalar_t>(), "tensorlib_addmv_naive");

  return result;
}

}} // namespace at::native
