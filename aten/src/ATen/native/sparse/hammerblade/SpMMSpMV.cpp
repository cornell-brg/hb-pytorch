#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

IntTensor _to_csr_int(const IntTensor& rowIndices, int64_t dim, int64_t nnz) {

  TORCH_CHECK(rowIndices.is_hammerblade(), "row Indices should be on hammerblade");

  IntTensor csr = at::zeros({dim + 1}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});

  uint32_t dim_uint = (uint32_t)(dim);
  uint32_t nnz_uint = (uint32_t)(nnz);
  hb_offload_kernel(csr, rowIndices, dim_uint, nnz_uint, "tensorlib_coo_to_csr");
}

void _to_c2sr(const IntTensor& rowIndices, 
              IntTensor& csr,
              IntTensor& c2sr,
              IntTensor& colindices,
              IntTensor& c2sr_colindices,
              Tensor& values,
              Tensor& c2sr_values,
              int64_t dim,
              int64_t nnz) {

  int dim_int = (int)(dim);
  int nnz_int = (int)(nnz);
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
  device_args.push_back(create_device_tensor(rowIndices, device_ptrs));
  device_args.push_back(create_device_tensor(csr, device_ptrs));
  device_args.push_back(create_device_tensor(c2sr, device_ptrs));
  device_args.push_back(create_device_tensor(colindices, device_ptrs));
  device_args.push_back(create_device_tensor(c2sr_colindices, device_ptrs));
  device_args.push_back(create_device_tensor(values, device_ptrs));
  device_args.push_back(create_device_tensor(c2sr_values, device_ptrs));
  device_args.push_back(create_device_scalar(dim_int));
  device_args.push_back(create_device_scalar(nnz_int));

  c10::hammerblade::offload_kernel("tensorlib_coo_to_c2sr", device_args);
  cleanup_device(device_args, device_ptrs);
}  

Tensor _sparse_mm_hb(const SparseTensor& sparse, const Tensor& dense) {

  TORCH_CHECK(sparse.is_hammerblade(), "SpMM: expected 'mat1' to be a HammerBlade tensor");
  TORCH_CHECK(dense.is_hammerblade(), "SpMM: expected 'mat2' to be a HammerBlade tensor");

  if ( (sparse.scalar_type() != ScalarType::Float)
    || (dense.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade SpMM is implemented for Float only"); 
  }
  
  using scalar_t = float;
 
  TORCH_CHECK(sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for HammerBlade sparse mm !");
  TORCH_CHECK(sparse.dim() == 2 && sparse.dim() == 2, "2D matrix expected, got ", sparse.dim(), " and ", dense.dim(), " tensors");
  TORCH_CHECK(sparse.size(1) == dense.size(0), "Argument #2: Expected dim 0 size ", sparse.size(1), ", got ", dense.size(0));

  int64_t nnz = sparse._nnz();
  int64_t dim = sparse.size(0);

  IntTensor indices = sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32");
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices show be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  IntTensor csr_hb = _to_csr_int(rowIndices, dim, nnz);
//  IntTensor csr_hb = at::empty({csr.size(0)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kInt)});
//  csr_hb.copy_(csr);
  
  Tensor values = sparse._values();

  Tensor result = at::zeros({sparse.size(0), dense.size(1)}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  hb_offload_kernel(result, csr_hb, colIndices, values, dense, "tensorlib_sparse_dense_mm");
  return result;
}

Tensor mv_hb_sparse(const SparseTensor& sparse, const Tensor& dense) {
  TORCH_CHECK(sparse.is_hammerblade(), "SpMV: expected 'mat1' to be a HammerBlade tensor");
  TORCH_CHECK(dense.is_hammerblade(), "SpMV: expected 'mat2' to be a HammerBlade tensor");

  if ( (sparse.scalar_type() != ScalarType::Float)
    || (dense.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade SpMV is implemented for Float only");
  }

  using scalar_t = float;

  TORCH_CHECK(sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for HammerBlade SpMV !");
  TORCH_CHECK(sparse.dim() == 2 && sparse.dim() == 2, "2D matrix expected, got ", sparse.dim(), " and ", dense.dim(), " tensors");
  TORCH_CHECK(dense.dim() == 1, "Argument #2: Expected vector, got dim", dense.dim()); 
  TORCH_CHECK(sparse.size(1) == dense.size(0), "Argument #2: Expected dim 0 size ", sparse.size(1), ", got ", dense.size(0));
  
  int64_t nnz = sparse._nnz();
  int64_t dim = sparse.size(0);
  int64_t dim1 = sparse.size(1); 
  int64_t estimate = dim * dim1;
 

  IntTensor indices = sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32");
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices show be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  Tensor values = sparse._values();

  IntTensor csr = at::empty({dim + 1}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  IntTensor c2sr = at::empty({2 * dim}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  IntTensor c2sr_colindices = at::empty({estimate}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  Tensor c2sr_values = at::empty({estimate}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  _to_c2sr(rowIndices, csr, c2sr, colIndices, c2sr_colindices, values, c2sr_values, dim, nnz);

  Tensor result = at::empty({dim}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  hb_offload_kernel(result, c2sr, c2sr_colindices, c2sr_values, dense, "tensorlib_spmv");
  return result;
}   
}}
