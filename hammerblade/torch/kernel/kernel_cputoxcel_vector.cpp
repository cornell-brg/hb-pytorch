//==========================================================================================
//The kernel code of changing the data layout of dense vector from CPU to HB with SpMV Xcel
//11/07/2020 Zhongyuan Zhao(zz546@cornell.edu)
//==========================================================================================
#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_cputoxcel_vector(
    hb_tensor_t* _dense_vector,
    hb_tensor_t* _vector) {

    auto dense_vector = HBTensor<int>(_dense_vector);
    auto vector = HBTensor<int>(_vector);
    uint32_t col = dense_vector.numel();
    int* v_ptr = (int*)vector.data_ptr();
    int* dv_ptr = (int*)dense_vector.data_ptr();

    uint32_t cacheline_word = CACHELINE_BYTE / 4; 
    uint32_t max_region_b = (((col + NUM_PE - 1) / NUM_PE) + cacheline_word - 1) / cacheline_word;
      
    //Now we use the first row to copy the data now, NUM_PE must equal to 16
    uint32_t b_index = __bsg_id * cacheline_word;
    if(__bsg_id <= 16) {
      for(uint32_t i=0; i<max_region_b; i++) {
        for(uint32_t k=0; k < cacheline_word; k++){
          uint32_t address_b = i * cacheline_word * NUM_PE + k * NUM_PE + __bsg_id;
          if(address_b < col)
            v_ptr[b_index] = dv_ptr[address_b];
          else
            v_ptr[b_index] = 0;
          b_index++;
        }
        b_index = b_index + NUM_PE * cacheline_word - cacheline_word;
      } 
    }
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_cputoxcel_vector, hb_tensor_t*, hb_tensor_t*)
}
