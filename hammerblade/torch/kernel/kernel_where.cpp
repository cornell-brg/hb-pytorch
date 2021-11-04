//====================================================================
// where kernel
// 09/03/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_where_byte(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          hb_tensor_t* t3_p) {
    auto res = HBTensor<float>(t0_p);
    auto condition_ten = HBTensor<uint8_t>(t1_p);
    auto x_ten = HBTensor<float>(t2_p);
    auto y_ten = HBTensor<float>(t3_p);

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](uint8_t condition, float x, float y) {
        return condition ? x : y;
      },
      res, condition_ten, x_ten, y_ten);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }
  __attribute__ ((noinline))  int tensorlib_where_bool(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          hb_tensor_t* t3_p) {
    auto res = HBTensor<float>(t0_p);
    auto condition_ten = HBTensor<bool>(t1_p);
    auto x_ten = HBTensor<float>(t2_p);
    auto y_ten = HBTensor<float>(t3_p);

    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    hb_tiled_foreach(
      [](bool condition, float x, float y) {
        return condition ? x : y;
      },
      res, condition_ten, x_ten, y_ten);

    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_where_byte, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*)
  HB_EMUL_REG_KERNEL(tensorlib_where_bool, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*)
}
