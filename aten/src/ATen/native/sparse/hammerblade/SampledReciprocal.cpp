#include <ATen/ATen.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

SparseTensor& _sampled_reciprocal__hb(SparseTensor& self) {
  TORCH_CHECK(self.is_hammerblade(), "_sampled_reciprocal__hb: expected 'self' to be a HammerBlade tensor");
  TORCH_CHECK(self.is_coalesced(), "_sampled_reciprocal__hb only supports a coalesced tensor");
  if ( self.scalar_type() != ScalarType::Float ) {
    AT_ERROR("HammerBlade _sampled_reciprocal__hb is implemented for Float type only"); 
  }
  hb_offload_kernel(self._values(), "tensorlib_reciprocal_");
  return self;
}

}}
