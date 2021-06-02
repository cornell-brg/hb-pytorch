//========================================================================
// Element-wise add kernel
//========================================================================
// This is a very simple element-wise add kernel, but in reality it
// actually does an elementise multiply then add.
//
// Authors : Lin Cheng Bandhav Veluri
// Date    : 03/05/2020, 07/13/2020

#include <kernel_common.hpp>
#include <cstdint>

// emulate mulh

inline uint32_t mulh(uint32_t a, uint32_t b) {

  uint32_t ah = (a & 0xffff0000) >> 16;
  uint32_t al = (a & 0xffff);
  uint32_t bh = (b & 0xffff0000) >> 16;
  uint32_t bl = (b & 0xffff);

  uint32_t ahbh = ah * bh;
  uint32_t albl = al * bl;
  uint32_t albh = al * bh + bl * ah;
  uint32_t albh_carry = 0;

  if (albh < al * bh || albh < bl * ah) {
    albh_carry = 1;
  }

  uint32_t albh_h = albh >> 16;
  uint32_t albh_l = (albh & 0xffff) << 16;
  uint32_t total_l = albh_l + albl;
  uint32_t carry = 0;

  if (total_l < albh_l || total_l < albl) {
    carry = 1;
  }

  uint32_t _mulh = ahbh + albh_h + carry + (albh_carry << 16);

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

  hb_tiled_foreach(
    [alpha](float a, float b) {
      return a + alpha * b;
  },
  c, a, b);

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

  hb_tiled_foreach(
    [alpha](long long a, long long b) {
      uint32_t ah = alpha >> 32;
      uint32_t al = alpha & 0xffffffff;
      uint32_t bh = b >> 32;
      uint32_t bl = b & 0xffffffff;
      uint32_t albl = al * bl;
      int64_t mul = ((int64_t)(ah * bl + al * bh) << 32) + (int64_t)(((uint64_t)mulh(al,bl)) << 32) + albl;
      return mul + a;
  },
  c, a, b);

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

  hb_tiled_foreach(
    [alpha](int a, int b) {
      return a + alpha * b;
  },
  c, a, b);

  g_barrier.sync();
  return 0;
}

HB_EMUL_REG_KERNEL(tensorlib_add_Int, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int*)

} /* extern C */
