//=======================================================================
// Convert COO format sparse matrix to C2SR (Compressed Cyclic Sparse Row) format
// 08/12/2020 Zhongyuan Zhao (zz546@cornell.edu)
//=======================================================================

#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" { 

  __attribute__ ((noinline)) int tensorlib_coo_to_c2sr(
          hb_tensor_t* _rowIndices,
          hb_tensor_t* _csr,  // 2dim + 1
          hb_tensor_t* _c2sr, // dim
          hb_tensor_t* _colindices,  // nnz
          hb_tensor_t* _c2sr_colindices, // nnz 
          hb_tensor_t* _values, // nnz
          hb_tensor_t* _c2sr_values, // nnz
          uint32_t* _dim,
          uint32_t* _nnz) {

    auto csr = HBTensor<int>(_csr);
    auto c2sr = HBTensor<int>(_c2sr);
    auto rowindices = HBTensor<int>(_rowIndices);
    auto colindices = HBTensor<int>(_colindices);
    auto c2sr_colindices = HBTensor<int>(_c2sr_colindices);
    auto values = HBTensor<float>(_values);
    auto c2sr_values = HBTensor<float>(_c2sr_values);
    uint32_t dim = *_dim;
    uint32_t nnz = *_nnz;
    
    uint32_t num_element = c2sr_values.numel();
    size_t thread_num = bsg_tiles_X * bsg_tiles_Y;
    size_t start = __bsg_id;
    size_t end = nnz;

    bsg_cuda_print_stat_kernel_start();
    if(__bsg_id == 0) {
      csr(0) = 0;
    }
    g_barrier.sync();

    // Generate CSR
    int h, hp0, hp1;
    for (size_t i = start; i < end; i = i + thread_num) {
      hp0 = rowindices(i);
      hp1 = (i+1 == nnz) ? dim : rowindices(i+1);
      if(hp0 != hp1) for(h = hp0; h < hp1; h++) {
        csr(h+1) = i+1;
      }
    }
    
    g_barrier.sync();
    
    //Generate nnz of each row, store into c2sr(dim) ~ c2sr(2 * dim - 1)
    end = dim;
    for (size_t j = start; j < end; j = j + thread_num) {
      c2sr(dim + j) = csr(j + 1) - csr(j);
      printf("c2sr(%d + %d) is %d\n", dim, j, c2sr(dim + j));
    }
    
    printf("pass generate nnz of each row\n");
    g_barrier.sync();

    // Generate the pointer to the first nnz element of each row in corresponding slot, store into c2sr(0) ~ c2sr(dim - 1)
    for (size_t k = start; k < end; k = k + thread_num) {
      int sum = 0;
      if(k < NUM_OF_SLOTS) {
        sum = 0;
      } else {
        int t = k;
        int temp = t - NUM_OF_SLOTS;
        for (; t >= 0 ; t = t - NUM_OF_SLOTS) {
          sum = sum + c2sr(dim + t);     
        }
      }
      
      c2sr(k) = sum;
      printf("c2sr(%d) is %d\n", k, c2sr(k)); 
    }
    
    printf("pass generating the pointer of the fist nnz element of each row\n");
    g_barrier.sync();

    //Reorganize the data in colindices and values
    for(size_t l = start; l < end; l = l + thread_num) {
      printf("l is %d\n", l);
      for(int32_t m = c2sr(l), n = csr(l); m < c2sr(l) + c2sr(dim + l) && n < csr(l + 1); m++, n++) {
        int idx = convert_idx(m, dim, l);
        printf("Got m is %d, l is %d and idx is %d\n", m, l, idx);
        if(idx >= num_element){
          printf("m is %d and idx is %d, but total number of element in c2sr values is only %d\n", m, idx, num_element);
        }
        c2sr_colindices(idx) = colindices(n);
        printf("c2sr_colindices(%d) is %d\n", idx, c2sr_colindices(idx));
        c2sr_values(idx) = values(n);
      }
    }  
    
    printf("successful finish coo_to_c2sr");
    bsg_cuda_print_stat_kernel_end();   
    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_coo_to_c2sr, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, uint32_t*, uint32_t*)

}


