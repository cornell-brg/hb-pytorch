//========================================================================
// Element-wise add kernel
//========================================================================
// This is a very simple element-wise add kernel, but in reality it
// actually does an elementise multiply then add.
//
// Authors : Lin Cheng Bandhav Veluri
// Date    : 03/05/2020, 07/13/2020

#include <kernel_common.hpp>

// emulate mulh

inline int32_t mulh(int32_t a, int32_t b) {
  
  int16_t ah = (a & 0xffff0000) >> 16;
  int16_t al = (a & 0xffff);
  int16_t bh = (b & 0xffff0000) >> 16;
  int16_t bl = (b & 0xffff);
  
  int32_t ahbh = ah * bh;
  int32_t alhb_h = (al * bh + bl * ah) >> 16;
  int32_t alhb_l = ((al * bh + bl * ah) & 0xffff) << 16;
  int32_t total_l = alhb_l + al * bl;
  int32_t carry = 0;
  
  if (total_l - alhb_l != al * bl) {
    carry = 1;
  }
  
  int32_t _mulh = ahbh + alhb_h + carry;
  std::cout << _mulh << std::endl;
  return _mulh;
}

// As with all HB kernels, We wrap them with extern "C" to prevent name
// mangling.

extern "C" {

//------------------------------------------------------------------------
// tensorlib_add
//------------------------------------------------------------------------
// This is the default add kernel for tensors with float elements.

__attribute__ ((noinline))
int tensorlib_add( hb_tensor_t* t0_p, hb_tensor_t* t1_p,
                   hb_tensor_t* t2_p, float* alpha_p)
{
  HBTensor<float> c(t0_p);
  HBTensor<float> a(t1_p);
  HBTensor<float> b(t2_p);
  float alpha = *alpha_p;

  bsg_cuda_print_stat_kernel_start();

  hb_tiled_foreach(
    [alpha](float a, float b) {
      return a + alpha * b;
  },
  c, a, b);

  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*)

//------------------------------------------------------------------------
// tensorlib_add_Long
//------------------------------------------------------------------------
// This is the add kernel for tensors with long long (64-bit) elements.
// It is critical that the kernel name end with Long to enable automatic
// dispatching from the host code.

__attribute__ ((noinline))
int tensorlib_add_Long( hb_tensor_t* t0_p, hb_tensor_t* t1_p,
                        hb_tensor_t* t2_p, long long* alpha_p)
{
  HBTensor<long long> c(t0_p);
  HBTensor<long long> a(t1_p);
  HBTensor<long long> b(t2_p);
  long long alpha = *alpha_p;

  bsg_cuda_print_stat_kernel_start();

  hb_tiled_foreach(
    [alpha](long long a, long long b) {
	  std::cout << "alpha: " << alpha << std::endl;
      int32_t ah = (alpha & 0xffff0000) >> 32;
      int32_t al = (alpha & 0xffff);
      int32_t bh = (b & 0xffff0000) >> 32;
      int32_t bl = (b & 0xffff);
      std::cout << ((ah * bl + al * bh) << 32 )<< std::endl;
      std::cout << ((ah * bl + al * bh) << 32) + mulh(al, bl) << std::endl;
      std::cout << ((ah * bl + al * bh) << 32) + mulh(al, bl) + al * bl + a << std::endl;
      return ((ah * bl + al * bh) << 32) + mulh(al, bl) + al * bl + a;
  },
  c, a, b);

  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_add_Long, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, long long*)

//------------------------------------------------------------------------
// tensorlib_add_Int
//------------------------------------------------------------------------
// This is the add kernel for tensors with int (32-bit) elements.
// It is critical that the kernel name end with Int to enable automatic
// dispatching from the host code.

__attribute__ ((noinline))
int tensorlib_add_Int( hb_tensor_t* t0_p, hb_tensor_t* t1_p,
                       hb_tensor_t* t2_p, int* alpha_p)
{
  HBTensor<int> c(t0_p);
  HBTensor<int> a(t1_p);
  HBTensor<int> b(t2_p);
  int alpha = *alpha_p;

  bsg_cuda_print_stat_kernel_start();

  hb_tiled_foreach(
    [alpha](int a, int b) {
      return a + alpha * b;
  },
  c, a, b);

  bsg_cuda_print_stat_kernel_end();
  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_add_Int, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int*)

} /* extern C */
