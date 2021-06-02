#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/hammerblade/SparseCommon.hpp>
#include <ATen/native/hammerblade/Offload.h>
#include <cmath>

namespace at { namespace native {
using namespace at::sparse; 

IntTensor to_csr(const int32_t* indices, int32_t dim, int32_t nnz) {
  IntTensor csr = native::zeros({dim + 1}, kInt);
  if (nnz > 0) {
    auto csr_accessor = csr.accessor<int32_t, 1>();
    at::parallel_for(0, nnz, 10000, [&](int32_t start, int32_t end) {
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

Tensor to_spmvxcel_format_cpu(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "2D matrix expected, got ", self.dim(), " tensor");
  auto indices = self._indices();
  auto values  = self.values();
  auto int_indices = indices.to(kInt);
  auto int_values = values.to(kInt);
  
  const uint32_t cacheline_word = CACHELINE_BYTE / 4;
  const uint32_t cacheline_log  = std::log(CACHELINE_BYTE) / std::log(2);
  const uint32_t num_pe_log = std::log(NUM_PE) / std::log(2);

  int32_t row = (int32_t)self.size(0);
  int32_t col = (int32_t)self.size(1);
  int32_t nnz = (int32_t)self._nnz();
  IntTensor csr = to_csr(int_indices.data_ptr<int32_t>(), row, nnz);
  int32_t* csr_ptr = csr.data_ptr<int32_t>();
  int32_t* csr_idx = int_indices.data_ptr<int32_t>();
  int32_t* csr_val = int_values.data_ptr<int32_t>();

  int32_t num_tile_x = std::ceil((float)col / (float)TILE_X_SIZE);
  int32_t num_tile_y = std::ceil((float)row / (float)TILE_Y_SIZE);
  int32_t last_tile_x = (col % TILE_X_SIZE == 0) ? TILE_X_SIZE : (col % TILE_X_SIZE);
  int32_t last_tile_y = (row % TILE_Y_SIZE == 0) ? TILE_Y_SIZE : (row % TILE_Y_SIZE);
//  std::cout << "last_tile_x is " << last_tile_x << std::endl;
  
  int32_t s_ptr_record_len = num_tile_x * num_tile_y;
  int32_t record_sparse_len[s_ptr_record_len];
//  Tensor record = at::empty({s_ptr_record_len}, {at::requires_grad().dtype(at::kInt)});
//  int32_t* record_ptr = record.data_ptr<int32_t>();
  int32_t len_sparse = 0;
  int32_t *tile_ptr[s_ptr_record_len];

  //Create a tensor that stores the other relative informations needed by the SpMV accelerators
 
  int32_t other_info[12];

  other_info[0] = row;
  other_info[1] = col;
  other_info[2] = num_tile_x;
  other_info[3] = last_tile_x;
  other_info[4] = num_tile_y;
  other_info[5] = last_tile_y;
  other_info[6] = TILE_X_SIZE;
  other_info[7] = TILE_Y_SIZE;
  other_info[8] = NUM_PE;
  other_info[9] = CACHELINE_BYTE;
  other_info[10] = cacheline_log;
  other_info[11] = num_pe_log;
/*
  Tensor other = at::empty({12}, {at::requires_grad().dtype(at::kInt)});
  int32_t* other_ptr = other.data_ptr<int32_t>();
  memcpy(other_ptr, other_info, 12 * sizeof(int32_t));
*/
//  std::cout << "Start tiling !" << std::endl;

  //loop to rebuild independent matrices for each tile
  for(uint32_t tile_y_idx = 0; tile_y_idx < num_tile_y; tile_y_idx++) {
    int tile_y_dim = (tile_y_idx == num_tile_y - 1) ? last_tile_y : TILE_Y_SIZE;
    int tile_x_bound[tile_y_dim];
    int tile_x_bound_last[tile_y_dim];
    for(uint32_t i = tile_y_idx * TILE_Y_SIZE; i < tile_y_idx * TILE_Y_SIZE + tile_y_dim; i++) {
      tile_x_bound[i % TILE_Y_SIZE] = csr_ptr[i];
      tile_x_bound_last[i % TILE_Y_SIZE] = csr_ptr[i];
    }
    for(uint32_t tile_x_idx = 0; tile_x_idx < num_tile_x; tile_x_idx++) {
      int sparse_length[tile_y_dim];
      for(uint32_t i = tile_y_idx * TILE_Y_SIZE; i < tile_y_idx * TILE_Y_SIZE + tile_y_dim; i++) {
        uint32_t row_len = 0;
        int upper_bound = (tile_x_idx + 1) *TILE_X_SIZE;
        for(uint32_t j = tile_x_bound[i % TILE_Y_SIZE]; j < csr_ptr[i + 1]; j++) {
          if(csr_idx[j] < upper_bound) {
            row_len++;
            tile_x_bound[i % TILE_Y_SIZE]++;
          } else {
            break;
          } 
        }
        sparse_length[i % TILE_Y_SIZE] = row_len;
      }
    
      //calculate content length of each PE, number is measured in elements/4bytes
      uint32_t len_per_pe_a[NUM_PE];
      for(int i = 0; i < NUM_PE; i++) {
        len_per_pe_a[i] = 0;
      }
      for(int i = 0; i < tile_y_dim; i++) {
        len_per_pe_a[i % NUM_PE] += 2 + sparse_length[i] * 2;
      }
    
      //the total size of the C2SR format tile is determined by the longest PE content
      uint32_t alloc_per_pe_a;
      uint32_t max_region_a = 0;
      for(int i=0; i<NUM_PE; i++){
        alloc_per_pe_a = ((len_per_pe_a[i] * 4) % (CACHELINE_BYTE) == 0) ? 
                         ((len_per_pe_a[i] * 4) / CACHELINE_BYTE) : (((len_per_pe_a[i] * 4)/CACHELINE_BYTE) + 1);
        if (alloc_per_pe_a > max_region_a){
          max_region_a = alloc_per_pe_a;
        }
      }
      uint32_t length_total_a = max_region_a * CACHELINE_BYTE * NUM_PE;

//      std::cout << "Enter the loop, create dram_a" << std::endl;
      //allocate memory space on the host for sparse matrix
      int32_t* dram_a = (int32_t*)malloc(length_total_a);
      uint32_t a_base = 0;
      int32_t* a_content[NUM_PE];
      for(uint32_t i=0; i<NUM_PE; i++) {
        a_content[i] = (int *) malloc(len_per_pe_a[i] * 4);
      }
      int32_t* dest[NUM_PE];
      for(uint32_t i=0; i<NUM_PE; i++){
        dest[i] = a_content[i];
      }
      for(uint32_t i=0; i<tile_y_dim; i++){
        *(dest[i%NUM_PE]) =  sparse_length[i];
        dest[i%NUM_PE] ++;
        *(dest[i%NUM_PE]) = a_base; //Loader will not actually use the ptr information in the sparse matrix, thus this is not the correct address
        dest[i%NUM_PE] ++;
      }
      for(uint32_t i = tile_y_idx * TILE_Y_SIZE; i < tile_y_idx * TILE_Y_SIZE + tile_y_dim; i++){
        for(uint32_t j = tile_x_bound_last[i%TILE_Y_SIZE]; j < tile_x_bound[i%TILE_Y_SIZE]; j++){
          *(dest[i%NUM_PE]) = csr_val[j]; 
          dest[i%NUM_PE] ++;   
          *(dest[i%NUM_PE]) = csr_idx[j];
          dest[i%NUM_PE] ++;  
        }
        tile_x_bound_last[i%TILE_Y_SIZE] = tile_x_bound[i%TILE_Y_SIZE];  
      }

      int index_a = 0;
      int num_loop = 0;
      int remain = 0;

      while(1){
        for(int i=0; i<NUM_PE; i++){
          for(int j=0; j<(CACHELINE_BYTE/(sizeof(int))); j++){
            if(len_per_pe_a[i]>0){
              dram_a[index_a] = *(a_content[i] + j + num_loop* (CACHELINE_BYTE / (sizeof(int))));
              len_per_pe_a[i]--;
              index_a++;
            } else{
              dram_a[index_a] = 0;
              index_a++;  
            }
          }
        }
        num_loop ++;
        for(int n=0; n<NUM_PE; n++){
          remain += len_per_pe_a[n];
        }
        if(remain == 0)
          break;
        else
          remain = 0;
      }    
    
      for(uint32_t i=0; i<NUM_PE; i++){
        free(a_content[i]);
      }

      record_sparse_len[tile_y_idx * num_tile_x + tile_x_idx] = length_total_a;
//      std::cout << "record_sparse_len[" << tile_y_idx * num_tile_x + tile_x_idx << "] is " << length_total_a << std::endl;
      len_sparse += length_total_a;
      tile_ptr[tile_y_idx * num_tile_x + tile_x_idx] = dram_a;
    }
  }

//  std::cout << "Exist the tiling loop, merge fragments" << std::endl;
//  std::cout << "c2sr length is " << len_sparse / sizeof(int32_t) << " record length is " << s_ptr_record_len << " other info length is 12" << std::endl; 
  Tensor c2sr_merge = at::empty({len_sparse/sizeof(int32_t) + s_ptr_record_len + 12}, {at::device(at::kCPU).dtype(at::kInt)});
  
  int32_t* src = c2sr_merge.data_ptr<int32_t>();
  for(uint32_t i = 0; i < s_ptr_record_len; i++){
    memcpy(src, tile_ptr[i],record_sparse_len[i]);
    free(tile_ptr[i]);
    src += record_sparse_len[i] / 4;
  }

  int32_t *record_ptr = c2sr_merge.data_ptr<int32_t>() + len_sparse/sizeof(int32_t);
  memcpy(record_ptr, record_sparse_len, s_ptr_record_len * sizeof(int32_t));
  int32_t *other_ptr = record_ptr + s_ptr_record_len;
  memcpy(other_ptr, other_info, 12 * sizeof(int32_t));
  return c2sr_merge;
}

Tensor to_spmmxcel_format_cpu(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "2D matrix expected, got ", self.dim(), " tensor");
  auto indices = self._indices();
  auto values  = self.values();
  auto int_indices = indices.to(kInt);
  auto int_values = values.to(kInt);

  int32_t row = (int32_t)self.size(0);
  int32_t col = (int32_t)self.size(1);
  int32_t nnz = (int32_t)self._nnz();  
  IntTensor csr = to_csr(int_indices.data_ptr<int32_t>(), row, nnz);
  int32_t* csr_ptr = csr.data_ptr<int32_t>();
  int32_t* csr_idx = int_indices.data_ptr<int32_t>();
  int32_t* csr_val = int_values.data_ptr<int32_t>();
  int32_t cacheline_word = CACHELINE_BYTE / 4;
  const uint32_t cacheline_log  = std::log(CACHELINE_BYTE) / std::log(2);
  const uint32_t num_pe_log = std::log(NUM_PE) / std::log(2);

  int32_t other_info[5];

  other_info[0] = row;
  other_info[1] = NUM_PE;
  other_info[2] = CACHELINE_BYTE;
  other_info[3] = cacheline_log;
  other_info[4] = num_pe_log;

//  std::cout << "m is " << row << "!" << std::endl;

  uint32_t len_per_pe_a[NUM_PE];
  for(int i = 0; i < NUM_PE; i++) {
    len_per_pe_a[i] = 0;
  }
  for(int i = 0; i < row; i++) {
    len_per_pe_a[i % NUM_PE] += 2 + (csr_ptr[i+1] - csr_ptr[i]) * 2;
  }

  uint32_t alloc_per_pe_a;
  uint32_t max_region_a = 0;
  for(int i=0; i<NUM_PE; i++){
    alloc_per_pe_a = (len_per_pe_a[i] + cacheline_word - 1) / cacheline_word;
    if (alloc_per_pe_a > max_region_a){
      max_region_a = alloc_per_pe_a;
    }
  }
 
//  std::cout << "Start to create dram_a" << std::endl;
  uint32_t length_total_a = max_region_a * CACHELINE_BYTE * NUM_PE;
  int32_t* dram_a = (int32_t*)malloc(length_total_a);
  uint32_t a_base = 0;
  int32_t* a_content[NUM_PE]; 

  for(uint32_t i = 0; i < NUM_PE; i++) {
    a_content[i] = (int *) malloc(len_per_pe_a[i] * 4);
  }
  int32_t* dest[NUM_PE];
  for(int32_t i=0; i<NUM_PE; i++){
    dest[i] = a_content[i];
  }
  
  for(uint32_t i=0; i<row; i++){
    *(dest[i%NUM_PE]) = csr_ptr[i+1] - csr_ptr[i];
    dest[i%NUM_PE]++;
    *(dest[i%NUM_PE]) = a_base; //Loader will not actually use the ptr information in the sparse matrix, thus this is not the correct address
    dest[i%NUM_PE] ++;
  }  

  for(uint32_t i=0; i<row; i++) {
    for(uint32_t j = csr_ptr[i]; j < csr_ptr[i+1]; j++) {
      *(dest[i%NUM_PE]) = csr_val[j];
      dest[i%NUM_PE] ++;
      *(dest[i%NUM_PE]) = csr_idx[j];
      dest[i%NUM_PE] ++;     
    }
  }
  
  int index_a = 0;
  int num_loop = 0;
  int remain = 0; 

  while(1){
    for(int i=0; i<NUM_PE; i++){
      for(int j=0; j<(CACHELINE_BYTE/(sizeof(int))); j++){
        if(len_per_pe_a[i]>0){
          dram_a[index_a] = *(a_content[i] + j + num_loop* (CACHELINE_BYTE / (sizeof(int))));
          len_per_pe_a[i]--;
          index_a++;
        } else{
          dram_a[index_a] = 0;
          index_a++;
        }
      }
    }
    num_loop ++;
    for(int n=0; n<NUM_PE; n++){
      remain += len_per_pe_a[n];
    }
    if(remain == 0)
      break;
    else
      remain = 0;
  }
  
  std::cout << "Finish write dram_a" << std::endl;
  for(uint32_t i = 0; i < NUM_PE; i++){
    free(a_content[i]);
  }
  
  std::cout << "Finish free " << std::endl;
  Tensor c2sr_merge = at::empty({length_total_a/sizeof(int32_t) + 5}, {at::device(at::kCPU).dtype(at::kInt)});
  int32_t* src = c2sr_merge.data_ptr<int32_t>();
  memcpy(src, dram_a, length_total_a);
  std::cout << "Finish copying c2sr" << std::endl;
  int32_t* other_ptr = src + length_total_a / sizeof(int32_t);
  memcpy(other_ptr, other_info, 5 * sizeof(int32_t));
  return c2sr_merge;
}

}} // namespace at::native   
