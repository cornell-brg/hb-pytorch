//====================================================================
//// tanh kernel
//// 05/07/2020 Jack Weber (jlw422@cornell.edu)
////====================================================================

#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_tanh(
          hb_tensor_t* _self) {

  auto self = HBTensor<float>(_self);
 
  // Start profiling
  bsg_cuda_print_stat_kernel_start();

  hb_parallel_for(self.numel(), [&](size_t i) {
    float a = tanh(self(i));
    self(i) = a; 
  }); 

  //end profiling
  bsg_cuda_print_stat_kernel_end();
  return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_tanh, hb_tensor_t*)

}

extern "C" {

  __attribute__ ((noinline))  int tensorlib_tanh_out(
         hb_tensor_t* _self, hb_tensor_t* _out) {

  auto self = HBTensor<float>(_self);
  auto out = HBTensor<float>(_out);

  // Start profiling
  bsg_cuda_print_stat_kernel_start();
 
  hb_parallel_for(self.numel(), [&](size_t i) {
    float a = tanh(self(i));
    out(i) = a;
  });

  //end profiling
  bsg_cuda_print_stat_kernel_end();
  return 0;
  }

    HB_EMUL_REG_KERNEL(tensorlib_tanh_out, hb_tensor_t*, hb_tensor_t*)
}                         
