//========================================================================
// Element-wise add kernel
//========================================================================
// This is a very simple element-wise add kernel, but in reality it
// actually does an elementise multiply then add.
//
// Authors : Lin Cheng Bandhav Veluri 
// Date    : 03/05/2020, 07/13/2020

#include <kernel_common.hpp>

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
      return a + alpha * b;
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
