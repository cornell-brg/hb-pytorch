//====================================================================
// _cat kernel
// 07/29/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib__cat(
       hb_tensor_t** tensors_p,
       hb_tensor_t* result_p,
       uint32_t* length_p,
       int32_t* dim_p) {

    if (__bsg_id == 0) {

      uint32_t length = *length_p;
      int32_t dim = *dim_p;

      std::cout << "_cat kernel TensorList length = " << length
                << " concat dim = " << dim << std::endl;

      std::cout << "hb_tensor_t** = " << tensors_p << std::endl;
      for (size_t i = 0; i < length; i++) {
        std::cout << " tensor at " << tensors_p[i] << std::endl;
        HBTensor<float> tensor(tensors_p[i]);
        std::cout << "tensor N=" << tensor.numel() << std::endl;
      }

      HBTensor<float> result(result_p);
      std::cout << "result N=" << result.numel() << std::endl;

    }

    g_barrier.sync();
    return 0;

  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib__cat, hb_tensor_t**, hb_tensor_t*, uint32_t*, int32_t*)

}
