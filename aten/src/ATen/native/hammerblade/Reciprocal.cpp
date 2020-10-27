#include <ATen/ATen.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

Tensor& _reciprocal__hb(Tensor& self) {
  TORCH_CHECK(self.is_hammerblade(), "_reciprocal__hb: expected 'self' to be a HammerBlade tensor");
  if ( self.scalar_type() != ScalarType::Float ) {
    AT_ERROR("HammerBlade reciprocal_ is implemented for Float type only"); 
  }
  hb_offload_kernel(self, "tensorlib_reciprocal_");
  return self;
}
   
}}
