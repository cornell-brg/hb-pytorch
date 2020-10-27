#include <ATen/ATen.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

// computes (b@c.T), sampled by self
Tensor sddtmmd_hb(const SparseTensor& sample, const Tensor& b, const Tensor& c) {

  TORCH_CHECK(sample.is_hammerblade(), "Sddtmmd: expected 'sample' to be a HammerBlade tensor");
  TORCH_CHECK(b.is_hammerblade(), "Sddtmmd: expected 'b' to be a HammerBlade tensor");
  TORCH_CHECK(c.is_hammerblade(), "Sddtmmd: expected 'c' to be a HammerBlade tensor");

  if ( b.scalar_type() != ScalarType::Float
    || c.scalar_type() != ScalarType::Float ) {
    AT_ERROR("HammerBlade Sddtmmd is implemented for Float type only for matrices b and c"); 
  }
   
  TORCH_CHECK(sample.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'sample' in HammerBlade Sddtmmd!");
  TORCH_CHECK(b.dim() == 2 && c.dim() == 2, "Expected 2D matrixes for 'b' and 'c', but got ", b.dim(), " and ", c.dim(), " tensors");
  TORCH_CHECK(b.size(1) == c.size(1), "Matrix multiply dimension mismatch: 'b' dim 1 = ", b.size(1), ", 'c'.T dim 0 = ", c.size(1)); 
  TORCH_CHECK(b.size(0) == sample.size(0) && c.size(0) == sample.size(1),"Sddtmmd sample dimension mismatch: sample was shape ",sample.size(0)," by ",sample.size(1),", but (b@c.T) is shape ",b.size(0)," by ",c.size(0));

  IntTensor indices = sample._indices();
  TORCH_CHECK(indices.is_hammerblade(), "indices must be HammerBlade Tensor");
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32, but got ", indices.dtype());
  IntTensor rowIndices = indices.select(0, 0);
  TORCH_CHECK(rowIndices.is_hammerblade(), "rowIndices must be HammerBlade Tensor");
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices must be HammerBlade Tensor");

  Tensor result = at::zeros(sample.sizes(), {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  hb_offload_kernel(result, rowIndices, colIndices, b, c, "tensorlib_sddtmmd");

  return result;
}
   
}}
