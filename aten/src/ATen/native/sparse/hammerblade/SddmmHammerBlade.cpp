#include <ATen/ATen.h>
#include <ATen/templates/NativeFunctions.h>
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
  TORCH_CHECK(b.size(0) == c.size(1), "Matrix multiply dimension mismatch: 'mat1' dim 0 = ", b.size(0), ", 'mat2' dim 1 = ", c.size(1));
  
  IntTensor indices = sample._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32");
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices show be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  
  Tensor result = at::zeros({b.size(1), c.size(0)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  int64_t dot_prod_len = b.size(0);
  int64_t nnz = sample._nnz();

  hb_offload_kernel(result, colIndices, rowIndices, b, c, dot_prod_len, nnz, "tensorlib_sddmm");
  return result;
}
   
}}
