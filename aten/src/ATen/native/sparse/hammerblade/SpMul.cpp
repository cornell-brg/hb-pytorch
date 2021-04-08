#include <ATen/ATen.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

SparseTensor mul_sparse_hb(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(1, "entering kernel");
  TORCH_CHECK(self.is_hammerblade(), "mul_sparse_hb: expected 'self' to be a HammerBlade tensor");
  TORCH_CHECK(other.is_hammerblade(), "mul_sparse_hb: expected 'other' to be a HammerBlade tensor");
  TORCH_CHECK(self.is_coalesced(), "mul_sparse_hb only supports a coalesced tensor");
  TORCH_CHECK(other.is_coalesced(), "mul_sparse_hb only supports a coalesced tensor");

  if ( (self.scalar_type() != ScalarType::Float)
    || (other.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade SpMM is implemented for Float only"); 
  }

  TORCH_CHECK(self.sparse_dim() == 2, "mul_sparse_hb: We do not support hybrid sparse tensor (self)");
  TORCH_CHECK(other.sparse_dim() == 2, "mul_sparse_hb: We do not support hybrid sparse tensor (other)");
  TORCH_CHECK(self.size(0) == other.size(0), "mul_sparse_hb: matrix rows should match but got, self: ", self.size(0), ", other: ", other.size(0));
  TORCH_CHECK(self.size(1) == other.size(1), "mul_sparse_hb: matrix cols should match but got, self: ", self.size(1), ", other: ", other.size(1)); 
  TORCH_CHECK(self._nnz() == other._nnz(), "number of nnz expected be equal but got self: ", self._nnz(), ", other: ", other._nnz());

  IntTensor indices = self._indices();
  Tensor self_values = self._values();
  Tensor other_values = other._values();

  Tensor result_indices = at::empty({2, self._nnz()}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kInt)});
  Tensor result_vals = at::empty(self._nnz(), {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  hb_offload_kernel(result_indices, result_vals, indices,self_values,other_values, "tensorlib_sp_mul_");


  //Create HB sparse tensor (from SparseLLCopy):
  SparseTensor sparse_tensor = detail::make_tensor<SparseTensorImpl>(TensorTypeSet(TensorTypeId::SparseHammerBladeTensorId), result_vals.options().dtype());
  get_sparse_impl(sparse_tensor)->resize_(self.sparse_dim(), self.dense_dim(), self.sizes());
  get_sparse_impl(sparse_tensor)->set_indices_and_values_unsafe(result_indices, result_vals);
  if(self.is_coalesced()) {
    get_sparse_impl(sparse_tensor)->set_coalesced(true);
  }

  return sparse_tensor;
}

}}