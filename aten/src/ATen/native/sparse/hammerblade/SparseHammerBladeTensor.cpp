#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/SparseTensorUtils.h>
//#include <ATen/cuda/CUDAUtils.h>

namespace at { namespace native {

using namespace at::sparse;
/*
SparseTensor & add_out_sparse_hammerblade(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, Scalar value) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_hammerblade(r, t, src, value);
  }
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  TORCH_CHECK(t.is_hammerblade(), "add: expected 'self' to be HammerBlade, but got CPU");
  TORCH_CHECK(r.is_hammerblade(), "add: expected 'out' to be HammerBlade, but got CPU");
  TORCH_CHECK(src_.is_hammerblade(), "add: expected 'src' to be HammerBlade, but got CPU");
  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());  

  if(src._nnz() == 0) {
    AT_ERROR(" 'other' should not have zero nnz values because on device memcpy is currently not allowed for HammerBlade");
  }
  if(t._nnz() == 0) {
    AT_ERROR(" 'self' should not have zero nnz values because currently mul_out_sparse_scalar is not supported for HammerBlade");
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");
  return r; 
}
*/  
  
  

}} // namespace at::native
