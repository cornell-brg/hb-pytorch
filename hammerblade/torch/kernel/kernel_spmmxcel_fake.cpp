#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline)) int tensorlib_spmmxcel(
    hb_tensor_t* _result,
    hb_tensor_t* _c2sr_m,
    hb_tensor_t* _matrix,
    hb_tensor_t* _other_info,
    int32_t* _n,
    int32_t* _k) {

    auto other_info = (int*)HBTensor<int>(_other_info).data_ptr();

    int m = other_info[0];
    int n = *(_n);
    int k = *(_k);
//    printf("Size of SpMM is %d x %d x %d\n", m, n, k);


    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_spmmxcel, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int32_t*, int32_t*)
}
  
