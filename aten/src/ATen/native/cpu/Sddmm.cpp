namespace at {
namespace native {

template <typename scalar_t>
void inline sddmm_kernel(
    const Tensor& a_sparse,
    const Tensor& b_dense,
    const Tensor& c_dense,
    Tensor& out_tensor) {
    
    scalar_t* a_idx = a_sparse._indices().data_ptr();
    scalar_t* b = b_dense.data_ptr();
    scalar_t* c = c_dense.data_ptr();
    scalar_t* out = out_tensor.data_ptr();

    int dot_len = b_dense.size(1);
    for (int k = 0; k < a_sparse._nnz(); k++) {
      int ai = a_idx[0][k];
      int aj = a_idx[1][k];

      float dot_total = 0;
      for (int i = 0; i < dot_len)
        dot_total += b[ai][i] * c[i][aj];
        
      out[ai][aj] = dot_total;
    }
}

Tensor sddmm_cpu(
    const SparseTensor& a_sparse,
    const Tensor& b_dense,
    const Tensor& c_dense) {
  //TODO: eventually change out_dense to out_sparse (return type of SparseTensor not yet supported..)

  // is out mutable?
  Tensor out_dense = at::zeros(a_sparse.sizes(), new TensorOptions());
  AT_DISPATCH_ALL_TYPES(a_sparse.scalar_type(), "sddmm_cpu", [&] {
    sddmm_kernel<scalar_t>(a_sparse, b_dense, c_dense, out_dense);
  });

  return out;
}

} // namespace native
} // namespace at
