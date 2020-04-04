#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

#include <ATen/SparseTensorUtils.h>

namespace at { namespace native {

using namespace at::sparse;

IntTensor _to_csr_int(const Tensor& dense, const IntTensor& rowIndices, int64_t dim, int64_t nnz) {
  IntTensor csr = native::zeros({dim+1}, kInt);

  int32_t* indices = rowIndices.data_ptr<int32_t>();
  
  if (nnz > 0) {
    auto csr_accessor = csr.accessor<int32_t, 1>();
      // Convert the sparse matrix to CSR format
    at::parallel_for(0, nnz, 10000, [&](int64_t start, int64_t end) {
      int64_t h, hp0, hp1;
      for (auto i = start; i < end; i++) {
        hp0 = indices[i];
        hp1 = (i+1 == nnz) ?  dim : indices[i+1];
        if (hp0 != hp1) for (h = hp0; h < hp1; h++) {
          csr_accessor[h+1] = i+1;
        }
      }
    });
  }

  return csr;
}

//We do not going to support csr for element-wise sparse tensor operation
Tensor& add_out_dense_sparse_hb(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, Scalar value) {
  AT_ASSERT(!r.is_sparse());
  AT_ASSERT(!dense.is_sparse());
  AT_ASSERT(sparse_.is_sparse());
  TORCH_CHECK(dense.is_hammerblade(), "add: expected 'self' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(sparse_.is_hammerblade(), "add: expected 'other' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(r.is_hammerblade(), "add: expected 'out' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(sparse_.is_coalesced(), "add: expected 'other' to be a coalesced tensor, but got an uncoalesced tensor");
  TORCH_CHECK(dense.sizes().equals(sparse_.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ", dense.sizes(), " while other has size ", sparse_.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");
  TORCH_CHECK(sparse_._indices().dtype() == at::kInt, "The Data type of the Sparse Tensor on HammerBlade should be Int32 !");
  IntTensor indices = sparse_._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Data type of indices on HB must be int32 !");
  if(r.numel() != dense.numel()) {
    AT_ERROR("Tensor size mismatch, got result=", r.numel(), "dense=", dense.numel());
  }
  if (!is_same_tensor(r, dense)) r.copy_(dense);
//  int64_t num_of_rows = sparse_.size(0);
//  int64_t nnz = sparse_.nnz();
//  IntTensor row_indices = indices.select(0, 0);
//  IntTensor col_indices = indices.select(0, 1);
//  IntTensor csr = _to_csr_int(const IntTensor& row_indices, num_of_rows, nnz);
  Tensor values = sparse_._values();
  hb_offload_kernel(r, dense, indices, values, "tensorlib_dense_sparse_add");
  return r;
}

SparseTensor & add_out_sparse_hb(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, Scalar value) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_hb(r, t, src, value);
  }
  else {
    AT_ERROR("Currently, only dense tensor add sparse tensor is supported on HammerBlade !");
  }
}
/*
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  TORCH_CHECK(t.is_hammerblade(), "add: expected 'self' to be HammerBlade, but got CPU");
  TORCH_CHECK(r.is_hammerblade(), "add: expected 'out' to be HammerBlade, but got CPU");
  TORCH_CHECK(src_.is_hammerblade(), "add: expected 'other' to be HammerBlade, but got CPU");
  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());  

  if(src._nnz() == 0) {
    AT_ERROR(" 'other' should not have zero nnz values because on device memcpy is currently not allowed for HammerBlade");
  }
  if(t._nnz() == 0) {
    AT_ERROR(" 'self' should not have zero nnz values because currently mul_out_sparse_scalar is not supported for HammerBlade");
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");
  
  return r;
*/ 
}} // namespace at::native
