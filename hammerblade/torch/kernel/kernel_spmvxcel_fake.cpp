#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_spmvxcel(
    hb_tensor_t* _result,
    hb_tensor_t* _c2sr_m,
    hb_tensor_t* _vector,
    hb_tensor_t* _len_record,
    hb_tensor_t* _other_info) {

    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_spmvxcel, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}
  
