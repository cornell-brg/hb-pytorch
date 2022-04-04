//=======================================================================
// Convert COO format sparse matrix to CBSR (Cyclic Bank Sparse Row) format
// 08/12/2020 Zhongyuan Zhao (zz546@cornell.edu)
//=======================================================================

#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" { 

  __attribute__ ((noinline)) int tensorlib_coo_to_c2sr(
          hb_tensor_t* _rowIndices,
          hb_tensor_t* _c2sr, // 2 * dim + 1
          hb_tensor_t* _colindices,  // nnz
          hb_tensor_t* _c2sr_colindices, // nnz 
          hb_tensor_t* _values, // nnz
          hb_tensor_t* _c2sr_values, // nnz
          uint32_t* _dim,
          uint32_t* _nnz) {

    auto c2sr = HBTensor<int>(_c2sr);
    auto rowindices = HBTensor<int>(_rowIndices);
    auto colindices = HBTensor<int>(_colindices);
    auto c2sr_colindices = HBTensor<int>(_c2sr_colindices);
    auto values = HBTensor<float>(_values);
    auto c2sr_values = HBTensor<float>(_c2sr_values);
    uint32_t dim = *_dim;
    uint32_t nnz = *_nnz;
    int *tmp_indices = (int*)c2sr_colindices.data_ptr();
    float *tmp_values = (float*)c2sr_values.data_ptr();
    int indices_vcache = ((uintptr_t)tmp_indices / (CACHE_LINE*4)) % NUM_OF_SLOTS;
    int values_vcache = ((uintptr_t)tmp_values / (CACHE_LINE*4)) % NUM_OF_SLOTS;
    int indices_padding = (NUM_OF_SLOTS - indices_vcache) * CACHE_LINE;
    int values_padding = (NUM_OF_SLOTS - values_vcache) * CACHE_LINE;
    
    uint32_t num_element = c2sr_values.numel();
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = nnz;
    uint32_t tag0 = 0;
//    uint32_t tag1 = 1;
//    uint32_t tag2 = 2;
    int32_t offset = dim + 1;

    bsg_cuda_print_stat_kernel_start();
    bsg_cuda_print_stat_start(tag0);
    if(__bsg_id == 0) {
      c2sr(0) = 0;
    }

    // Generate CSR
    int h, hp0, hp1;
    for (size_t i = start; i < end; i = i + thread_num) {
      hp0 = rowindices(i);
      hp1 = (i+1 == nnz) ? dim : rowindices(i+1);
      if(hp0 != hp1) for(h = hp0; h < hp1; h++) {
        c2sr(h+1) = i+1;
      }
    }
    
    g_barrier.sync(); 
    //Generate nnz of each row, store into c2sr(dim + 1) ~ c2sr(2 * dim)
    end = dim;
    // Generate the pointer to the first nnz element of each row in corresponding slot, store into c2sr(0) ~ c2sr(dim - 1)
    if(__bsg_id == 0) { 
      int first_end = dim > NUM_OF_SLOTS ? NUM_OF_SLOTS : dim;  
      for( size_t k = 0; k < first_end; k++) {
        c2sr(offset + k) = 0;
      } 

      for (size_t k = NUM_OF_SLOTS; k < end; k++) {
        int t = k - NUM_OF_SLOTS;
        c2sr(offset + k) = c2sr(offset + t) + c2sr(t + 1) - c2sr(t);     
      }
    }
    g_barrier.sync();
//    bsg_cuda_print_stat_end(tag1);

//    bsg_cuda_print_stat_start(tag2);
    //Reorganize the data in colindices and values
    for(size_t l = start; l < end; l = l + thread_num) {
//      bsg_printf("l is %d\n", l);
      int32_t c2sr_first = c2sr(offset + l);
      int32_t c2sr_last = c2sr(offset + l) + c2sr(l + 1) - c2sr(l);
      int32_t csr_first = c2sr(l);
      int32_t csr_last = c2sr(l + 1);
      for(int32_t m = c2sr_first, n = csr_first; m < c2sr_last && n < csr_last; m++, n++) {
        int idx = convert_idx(m, dim, l);
        int idx_indices = idx + indices_padding;
        int idx_values = idx + values_padding;
//        printf("Got m is %d, n is %d and idx is %d\n", m, n, idx);
        c2sr_colindices(idx_indices) = colindices(n);
//        printf("c2sr_colindices(%d) is %d\n", idx, c2sr_colindices(idx));
        c2sr_values(idx_values) = values(n);
      }
    }  
    
//    printf("successful finish coo_to_c2sr");
    bsg_cuda_print_stat_end(tag0);
    bsg_cuda_print_stat_kernel_end();   
    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_coo_to_c2sr, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, uint32_t*, uint32_t*)

}


