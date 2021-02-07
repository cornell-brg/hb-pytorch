//====================================================================
// copy_hb_to_hb kernel
// 03/18/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================
// Can't call memcpy directly here. Since the src tensor's stride could
// be zero.

#include <kernel_common.hpp>

  template<typename TS, typename TD>
  int tensorlib_copy_impl(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p) {
    auto res = HBTensor<TD>(t0_p);
    auto input = HBTensor<TS>(t1_p);

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach_conversion(res, input,
      [](TS a) {
        return static_cast<TD>(a);
    });

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;

  }

// We wrap all external-facing C++ kernels with `extern "C"` to
// prevent name mangling

extern "C" {

  __attribute__ ((noinline))  int tensorlib_copy_Int_to_Int(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<int,int>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Int_to_Int, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Int_to_Long(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<int,long long>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Int_to_Long, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Int_to_Float(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<int,float>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Int_to_Float, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Int_to_Double(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<int,double>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Int_to_Double, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Long_to_Int(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<long long,int>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Long_to_Int, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Long_to_Long(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<long long,long long>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Long_to_Long, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Long_to_Float(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<long long,float>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Long_to_Float, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Long_to_Double(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<long long,double>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Long_to_Double, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Float_to_Int(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<float,int>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Float_to_Int, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Float_to_Long(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<float,long long>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Float_to_Long, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Float_to_Float(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<float,float>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Float_to_Float, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Float_to_Double(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<float,double>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Float_to_Double, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Double_to_Int(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<double,int>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Double_to_Int, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Double_to_Long(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<double,long long>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Double_to_Long, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Double_to_Float(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<double,float>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Double_to_Float, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_copy_Double_to_Double(
              hb_tensor_t* t0_p,
              hb_tensor_t* t1_p) {
    return tensorlib_copy_impl<double,double>(t0_p, t1_p);
  }

  HB_EMUL_REG_KERNEL(tensorlib_copy_Double_to_Double, hb_tensor_t*, hb_tensor_t*)

}
