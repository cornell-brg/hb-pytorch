#include <ATen/ATen.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

/* 
- A.Transpose for a sparse matrix is the same as swapping rowIndices and colIndices (then sorting v, c, and r by rowIndices)
- csc(A) = csr(A.Transpose)
==> csc(A) = csr('rowIndices' = A.T's rowIndices (!= A's colIndices since they're not sorted)).
*/
Tensor _to_csc(const IntTensor& aTrowIndices, int64_t dim, int64_t nnz) {
    Tensor csc = at::zeros({dim + 1}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});
    hb_offload_kernel(csc, aTrowIndices, dim, nnz, "tensorlib_coo_to_csr");
    return csc;
}

Tensor dstmmt_hb(const Tensor& a_dense, const SparseTensor& bT_sparse) {

  TORCH_CHECK(a_dense.is_hammerblade(), "DenseSparseTMmT: expected 'a' to be a HammerBlade tensor");
  TORCH_CHECK(bT_sparse.is_hammerblade(), "DenseSparseTMmT: expected 'b' to be a HammerBlade tensor");

  if ( a_dense.scalar_type() != ScalarType::Float
    || bT_sparse.scalar_type() != ScalarType::Float ) {
    AT_ERROR("HammerBlade dstmmt is implemented for Float only"); 
  }
  TORCH_CHECK(bT_sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'b' (sparse) in HammerBlade DenseSparseTMmT!");
  TORCH_CHECK(a_dense.dim() == 2 && bT_sparse.dim() == 2, "Expected 2D matrixes for 'a' and 'b', but got ", a_dense.dim(), " and ", bT_sparse.dim(), " tensors");
  TORCH_CHECK(a_dense.size(1) == bT_sparse.size(1), "Matrix multiply dimension mismatch: 'a' dim 1 = ", a_dense.size(1), ", 'b' dim 0 = ", bT_sparse.size(1));
  
  IntTensor indices = bT_sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32, but got ", indices.dtype());
  IntTensor b_rowIndices = indices.select(0, 1); // = bT_colIndices
  TORCH_CHECK(b_rowIndices.is_hammerblade(), "b_rowIndices must be HammerBlade Tensor");
  IntTensor b_colIndices = indices.select(0, 0); // = bT_rowIndices
  TORCH_CHECK(b_colIndices.is_hammerblade(), "b_colIndices must be HammerBlade Tensor");
  IntTensor b_values = bT_sparse._values();

  Tensor result = at::zeros({bT_sparse.size(0), a_dense.size(0)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});
  int64_t b_nnz = bT_sparse._nnz();
  int64_t b_dim = bT_sparse.size(0);
  
  IntTensor b_csc = _to_csc(b_colIndices, b_dim, b_nnz);

  hb_offload_kernel(result, a_dense, b_csc, b_rowIndices, b_values, "tensorlib_dstmmt");
  return result;
}
   
}}
