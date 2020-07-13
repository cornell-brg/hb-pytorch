#include <ATen/ATen.h>
// #include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

// computes b@c only at the locations where sample is nonzero
Tensor sddmm_hb(const SparseTensor& sample, const Tensor& b, const Tensor& c) {

  TORCH_CHECK(sample.is_hammerblade(), "Sddmm: expected 'sample' to be a HammerBlade tensor");
  TORCH_CHECK(b.is_hammerblade(), "Sddmm: expected 'mat1' to be a HammerBlade tensor");
  TORCH_CHECK(c.is_hammerblade(), "Sddmm: expected 'mat2' to be a HammerBlade tensor");

  if ( sample.scalar_type() != ScalarType::Float
    || b.scalar_type() != ScalarType::Float
    || c.scalar_type() != ScalarType::Float ) {
    AT_ERROR("HammerBlade sddmm is implemented for Float only"); 
  }
   
  TORCH_CHECK(sample.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'sample' in HammerBlade sddmm!");
  TORCH_CHECK(b.dim() == 2 && c.dim() == 2, "Expected 2D matrixes for 'mat1' and 'mat2', but got ", b.dim(), " and ", c.dim(), " tensors");
  TORCH_CHECK(b.size(1) == c.size(0), "Matrix multiply dimension mismatch: 'mat1' dim 1 = ", b.size(1), ", 'mat2' dim 0 = ", c.size(0));
  
  IntTensor indices = sample._indices();
  if (!(indices.dtype() == at::kInt)) TORCH_WARN("Indices on HammerBlade should be int32");
  IntTensor colIndices = indices.select(0, 1);
  if (!colIndices.is_hammerblade()) TORCH_WARN("colIndices show be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  if (!rowIndices.is_hammerblade()) TORCH_WARN("rowIndices show be HammerBlade Tensor");
  
  Tensor result = at::zeros({b.size(0), c.size(1)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  int64_t dot_prod_len = b.size(1);
  int64_t nnz = sample._nnz();

  hb_offload_kernel(result, colIndices, rowIndices, b, c, dot_prod_len, nnz, "tensorlib_sddmm");
  return result;
}
   
}}
