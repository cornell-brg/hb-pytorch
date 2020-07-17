#include <ATen/ATen.h>
// #include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

// computes b@c only at the locations where sample is nonzero
Tensor dsmm_hb(const Tensor& a_dense, const SparseTensor& b_sparse) {

  TORCH_CHECK(a_dense.is_hammerblade(), "DenseSparseMm: expected 'a' to be a HammerBlade tensor");
  TORCH_CHECK(b_sparse.is_hammerblade(), "DenseSparseMm: expected 'b' to be a HammerBlade tensor");

  if ( a_dense.scalar_type() != ScalarType::Float
    || b_sparse.scalar_type() != ScalarType::Float ) {
    AT_ERROR("HammerBlade dsmp is implemented for Float only"); 
  }

  TORCH_CHECK(b_sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'b' (sparse) in HammerBlade DenseSparseMm!");
  TORCH_CHECK(a_dense.dim() == 2 && b_sparse.dim() == 2, "Expected 2D matrixes for 'a' and 'b', but got ", a_dense.dim(), " and ", b_sparse.dim(), " tensors");
  TORCH_CHECK(a_dense.size(1) == b_sparse.size(0), "Matrix multiply dimension mismatch: 'a' dim 1 = ", a_dense.size(1), ", 'b' dim 0 = ", b_sparse.size(0));
  
  IntTensor indices = b_sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32, but got ", indices.dtype());
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices must be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  TORCH_CHECK(rowIndices.is_hammerblade(), "rowIndices must be HammerBlade Tensor");
  IntTensor values = b_sparse._values();

  Tensor result = at::zeros({a_dense.size(0), b_sparse.size(1)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});
  int64_t b_nnz = b_sparse._nnz();
  int64_t b_dim = b_sparse.dim();
  
  hb_offload_kernel(result, a_dense, colIndices, rowIndices, values, "tensorlib_dsmm");
  return result;
}
   
}}
