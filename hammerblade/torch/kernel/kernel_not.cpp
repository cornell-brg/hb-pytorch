//========================================================================
// Element-wise not kernel
//========================================================================
//
// Authors : Janice Wei
// Date    : 10/08/2020
#include <kernel_common.hpp>
#include <cstdint>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_not(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    std::cout << "device code start";
    auto res = HBTensor<int>(t0_p);
    auto input = HBTensor<int>(t1_p);
    std::cout << "bsg_cuda_start";
    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [](int a) {
        std::cout << ~a;
        return ~a;
      },
      res, input);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_not, hb_tensor_t*, hb_tensor_t*)

}
