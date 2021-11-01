//============================================================================
// Pytorch C++ layer for SpMV SpMM SpMSpV and SpGeMM
// 04/05/2020 Zhongyuan Zhao (zz546@cornell.edu)
//===========================================================================
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

using namespace at::sparse;

IntTensor _to_csr_int(const IntTensor& rowIndices, int64_t dim, int64_t nnz) {

  TORCH_CHECK(rowIndices.is_hammerblade(), "row Indices should be on hammerblade");

  IntTensor csr = at::empty({dim + 1}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});

  uint32_t dim_uint = (uint32_t)(dim);
  uint32_t nnz_uint = (uint32_t)(nnz);
  hb_offload_kernel(csr, rowIndices, dim_uint, nnz_uint, "tensorlib_coo_to_csr");
  return csr;
}

void _to_c2sr(const IntTensor& rowIndices,
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

/*
Tensor _sparse_mm_hb(const SparseTensor& sparse, const Tensor& dense) {
  std::cout << "Already enter the _sparse_mm_hb function" << std::endl;
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
  std::cout << "Entering the spmm kernel" << std::endl;
  hb_offload_kernel(result, csr_hb, colIndices, values, dense, "tensorlib_spmm");
  return result;
}
*/

Tensor xcelspmv_hb(const SparseTensor& sparse, const Tensor& dense) {

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
  //When number of rows less than vcache number...
  dim1 = dim1 < 32 ? 32 : dim1;
  int64_t estimate = dim * dim1;
  //Consider the padding. There must be at least 1024 paddings...
  estimate = estimate + 1024;


  IntTensor indices = sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kInt, "Indices on HammerBlade should be int32");
  IntTensor colIndices = indices.select(0, 1);
  TORCH_CHECK(colIndices.is_hammerblade(), "colIndices show be HammerBlade Tensor");
  IntTensor rowIndices = indices.select(0, 0);
  Tensor values = sparse._values();

  IntTensor c2sr = at::empty({2 * dim + 1}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  IntTensor c2sr_colindices = at::empty({estimate}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  Tensor c2sr_values = at::empty({estimate}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  _to_c2sr(rowIndices, c2sr, colIndices, c2sr_colindices, values, c2sr_values, dim, nnz);

  Tensor result = at::empty({dim}, {at::requires_grad().device(at::kHAMMERBLADE).dtype(at::kFloat)});

  // create block (size can be 3 or 31)
  int block_size = 31;
  IntTensor block = at::empty({block_size * dim}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});

  hb_offload_kernel(block, c2sr, c2sr_colindices, c2sr_values, block_size, "tensorlib_block_generate");
  hb_offload_kernel(result, c2sr, c2sr_colindices, c2sr_values, dense, block, block_size, "tensorlib_spmv_SP1");

  //hb_offload_kernel(result, c2sr, c2sr_colindices, c2sr_values, dense, "tensorlib_spmv");
  return result;
}
}}


/*
//====================================================================
//The Pytorch C++ layer for xcelspmv operator
//10/30/2020 Zhongyuan Zhao (zz546@cornell.edu)
//=====================================================================
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>
#include <ATen/native/hammerblade/SparseCommon.hpp>
#include <chrono>

namespace at { namespace native {

Tensor xcelspmv_hb(const Tensor &self, const Tensor &dense_vector) {
  TORCH_CHECK(self.dim() == 1, "1D matrix expected, got dim ", self.dim(), " tensor");
  Tensor int_vector;
  if (!(dense_vector.dtype() == at::kInt)) {
    int_vector = dense_vector.to(at::kInt);
  } else {
    int_vector = dense_vector;
  }
  TORCH_CHECK(int_vector.dtype() == at::kInt, "Dense vector should be int !");
  TORCH_CHECK(self.dtype() == at::kInt, "C2SR should be int !");
  int64_t tensor_size = self.size(0);
  int32_t vector_size = (int32_t)int_vector.size(0);

  std::vector<int64_t> split_sizes(2);
  split_sizes[0] = tensor_size - 12;
  split_sizes[1] = 12;

  std::vector<Tensor> tensors = self.split_with_sizes(split_sizes, 0);
  Tensor other_hb = tensors[1];
  int64_t other_size = other_hb.size(0);

  Tensor other_cpu = other_hb.to(at::DeviceType::CPU);
  int32_t *other_info = other_cpu.data_ptr<int32_t>();

  int32_t row = other_info[0];
  int32_t col = other_info[1];
  TORCH_CHECK(col == vector_size, "Number of columns in sparse matrix must be equal to vector size !");
  int32_t num_tile_x = other_info[2];
  int32_t num_tile_y = other_info[4];
  int32_t record_len = num_tile_x * num_tile_y;

  Tensor c2srandrecord = tensors[0];
  int64_t c2sr_size = c2srandrecord.size(0);
 
  tensor_size = c2srandrecord.size(0);
  std::vector<int64_t> split_sizes1(2);
  split_sizes1[0] = tensor_size - record_len;
  split_sizes1[1] = record_len;
  std::vector<Tensor> tensors1 = c2srandrecord.split_with_sizes(split_sizes1, 0);
  
  Tensor c2sr_hb = tensors1[0];
  Tensor record_hb = tensors1[1];

//  uint32_t cacheline_word = CACHELINE_BYTE / 4;
//  uint32_t max_region_b = (((col + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
//  uint32_t length_total_b = max_region_b * CACHELINE_BYTE * NUM_PE;
//  std::cout << "Convert the CPU data layout of dense vector to SpMV Xcel" << std::endl;
//  Tensor vector_hb = at::empty({length_total_b / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)}); 
//  hb_offload_kernel(int_vector, vector_hb, "tensorlib_cputoxcel_vector");
//  std::cout << "Perform SpMV on Xcel" << std::endl;
//  uint32_t max_region_c = (((row + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
//  uint32_t length_total_c = max_region_c * CACHELINE_BYTE * NUM_PE;
//  std::cout << "Xcel_out is " << length_total_c << " and cpu out is " << row << std::endl;
//  Tensor xcel_out = at::empty({length_total_c / 4}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  Tensor result = at::empty({row}, {at::device(at::kHAMMERBLADE).dtype(at::kInt)});
  hb_offload_kernel(result, c2sr_hb, int_vector, record_hb, other_hb, "tensorlib_spmvxcel");
//  std::cout << "Convert the SpMV Xcel data layout of dense vector to CPU" << std::endl;
//  hb_offload_kernel(result, xcel_out, "tensorlib_xceltocpu_vector");

  return result;
}

}} 
*/
