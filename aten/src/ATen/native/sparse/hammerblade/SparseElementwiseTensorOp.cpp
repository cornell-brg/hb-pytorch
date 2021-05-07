//============================================================================
// Pytorch C++ layer for elementwise sparse tensor operators
// 03/22/2020 Zhongyuan Zhao (zz546@cornell.edu)
//============================================================================
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

#include <ATen/SparseTensorUtils.h>

namespace at { namespace native {

using namespace at::sparse;

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
  using scalar_t = float;
  hb_offload_kernel(r, dense, indices, values, value.to<scalar_t>(), "tensorlib_sparse_add_out_dense");
  return r;
}

SparseTensor& add_out_sparse_sparse_hb(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, Scalar value){
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(src.is_sparse());
  TORCH_CHECK(t.is_hammerblade(), "add: expected 't' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(src.is_hammerblade(), "add: expected 'src' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(r.is_hammerblade(), "add: expected 'out' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(t.sparse_dim() == t.dim(), "add: expected 't' to be a full dimension sparse tensor");
  TORCH_CHECK(t.sparse_dim() == src.dim(), "add: expected 'src' to be a full dimension sparse tensor");
  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 't' and 'src' to match, but ", t.sizes(), " != ", src.sizes());
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
  bool t_coalesced = t.is_coalesced(), s_coalesced = src.is_coalesced();
  IntTensor t_indices = t._indices();
  Tensor t_values = t._values();
  IntTensor src_indices = src._indices();
  Tensor src_values = src._values();
  r.resize_as_(src);
  IntTensor r_indices = at::empty({t.dim(), max_nnz}, t_indices.options());
  Tensor r_values = new_values_with_size_of(src_values, max_nnz).zero_();
  Tensor hb_resultNnz = at::empty({1}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
  using scalar_t = float;
  hb_offload_kernel(r_indices, r_values,  t_indices, t_values, src_indices, src_values, hb_resultNnz, value.to<scalar_t>(), "tensorlib_sparse_add_out_dense");
  IntTensor cpu_intresultNnz = at::empty({1}, CPU(kInt));
  cpu_intresultNnz.copy_(hb_resultNnz);
  LongTensor cpu_resultNnz = cpu_intresultNnz.to(kLong);
  get_sparse_impl(r) ->set_nnz_and_narrow(cpu_resultNnz.accessor<int64_t, 1>()[0]); 
  return r._coalesced_(t_coalesced && s_coalesced);
}

SparseTensor& mul_out_sparse_hb(SparseTensor& r, const SparseTensor& t, const SparseTensor& src){
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(src.is_sparse());
  TORCH_CHECK(t.is_hammerblade(), "mul: expected 't' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(src.is_hammerblade(), "mul: expected 'src' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(r.is_hammerblade(), "mul: expected 'out' to be a HAMMERBLADE tensor, but got a CPU tensor");
  TORCH_CHECK(t.sparse_dim() == t.dim(), "mul: expected 't' to be a full dimension sparse tensor");
  TORCH_CHECK(t.sparse_dim() == src.dim(), "mul: expected 'src' to be a full dimension sparse tensor");
  TORCH_CHECK(t.sizes().equals(src.sizes()), "mul: expected sizes of 't' and 'src' to match, but ", t.sizes(), " != ", src.sizes());
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);
  bool t_coalesced = t.is_coalesced(), s_coalesced = src.is_coalesced();
  IntTensor t_indices = t._indices();
  Tensor t_values = t._values();
  IntTensor src_indices = src._indices();
  Tensor src_values = src._values();
  r.resize_as_(src);
  IntTensor r_indices = at::empty({t.dim(), max_nnz}, t_indices.options());
  Tensor r_values = new_values_with_size_of(src_values, max_nnz).zero_();
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);
  Tensor hb_resultNnz = at::empty({1}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(r_indices, r_values,  t_indices, t_values, src_indices, src_values, hb_resultNnz, "tensorlib_sparse_mul");
  IntTensor cpu_intresultNnz = at::empty({1}, CPU(kInt));
  cpu_intresultNnz.copy_(hb_resultNnz);
  LongTensor cpu_resultNnz = cpu_intresultNnz.to(kLong);
  get_sparse_impl(r) ->set_nnz_and_narrow(cpu_resultNnz.accessor<int64_t, 1>()[0]);
  return r._coalesced_(t_coalesced && s_coalesced);
}

SparseTensor& add_out_sparse_hb(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, Scalar value) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_hb(r, t, src, value);
  }
  else {
    add_out_sparse_sparse_hb(r, t, src, value);
  }
}
/*
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  TORCH_CHECK(t.is_hammerblade(), "add: expected 'self' to be HammerBlade, but got CPU");
  TORCH_CHECK(r.is_hammerblade(), "add: expected 'out' to be HammerBlade, but got CPU");
  TORCH_CHECK(src_.is_hammerblade(), "add: expected 'other' to be HammerBlade, but got CPU");
  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());  

    AT_ERROR(" 'other' should not have zero nnz values because on device memcpy is currently not allowed for HammerBlade");
  }
  if(t._nnz() == 0) {
    AT_ERROR(" 'self' should not have zero nnz values because currently mul_out_sparse_scalar is not supported for HammerBlade");
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");
  
  return r;
*/ 
}} // namespace at::native
