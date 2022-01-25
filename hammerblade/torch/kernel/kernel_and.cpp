//========================================================================
// Element-wise and kernel
//========================================================================
//
// Authors : Janice Wei
// Date    : 09/25/2020

#include <kernel_common.hpp>
#include <cstdint>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_and_int(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    auto res = HBTensor<int>(t0_p);
    auto input1 = HBTensor<int>(t1_p);
    auto input2 = HBTensor<int>(t2_p);

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [](int a, int b) {
        return a & b;
      },
      res, input1, input2);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_and_int, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

  __attribute__ ((noinline))  bool tensorlib_and_bool(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    auto res = HBTensor<bool>(t0_p);
    auto input1 = HBTensor<bool>(t1_p);
    auto input2 = HBTensor<bool>(t2_p);

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(
      [](bool a, bool b) {
        return a & b;
      },
      res, input1, input2);

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_and_bool, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
