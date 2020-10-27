#include <ATen/ATen.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

// computes (b@c.T), sampled by self
SparseTensor sddtmm_hb(const SparseTensor& sample, const Tensor& b, const Tensor& c) {

  TORCH_CHECK(sample.is_hammerblade(), "SddTmm: expected 'sample' to be a HammerBlade tensor");
  TORCH_CHECK(b.is_hammerblade(), "SddTmm: expected 'b' to be a HammerBlade tensor");
  TORCH_CHECK(c.is_hammerblade(), "SddTmm: expected 'c' to be a HammerBlade tensor");

  if ( b.scalar_type() != ScalarType::Float
    || c.scalar_type() != ScalarType::Float ) {
    AT_ERROR("HammerBlade SddTmm is implemented for Float type only for matrices b and c"); 
  }
   
  TORCH_CHECK(sample.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'sample' in HammerBlade SddTmm!");
  TORCH_CHECK(b.dim() == 2 && c.dim() == 2, "Expected 2D matrixes for 'b' and 'c', but got ", b.dim(), " and ", c.dim(), " tensors");
  TORCH_CHECK(b.size(1) == c.size(1), "Matrix multiply dimension mismatch: 'b' dim 1 = ", b.size(1), ", 'c'.T dim 0 = ", c.size(1)); 
  
  IntTensor indices = sample._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32, but got ", indices.dtype());
  TORCH_CHECK(indices.is_hammerblade(), "indices must be HammerBlade Tensor");
  TORCH_CHECK(b.size(0) == sample.size(0) && c.size(0) == sample.size(1),"SddTmm sample dimension mismatch: sample was shape ",sample.size(0)," by ",sample.size(1),", but (b@c.T) is shape ",b.size(0)," by ",c.size(0));

  Tensor result_indices = at::zeros({2, sample._nnz()}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kInt)});
  Tensor result_vals = at::zeros(sample._nnz(), {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  hb_offload_kernel(result_indices, result_vals, indices, b, c, "tensorlib_sddtmm");

  //Create HB sparse tensor (from SparseLLCopy):
  SparseTensor sparse_tensor = detail::make_tensor<SparseTensorImpl>(TensorTypeSet(TensorTypeId::SparseHammerBladeTensorId), result_vals.options().dtype());
  get_sparse_impl(sparse_tensor)->resize_(sample.sparse_dim(), sample.dense_dim(), sample.sizes());
  get_sparse_impl(sparse_tensor)->set_indices_and_values_unsafe(result_indices, result_vals);
  if(sample.is_coalesced()) {
    get_sparse_impl(sparse_tensor)->set_coalesced(true);
  }

  return sparse_tensor;
}
   
}}
