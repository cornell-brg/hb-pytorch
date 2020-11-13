//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#define TILE_LOAD_RANGE 512 // 1024 * 128 * 4 = 128 * 8 * 16 * 32
#define LOAD_ITER 3

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_load_mbmark(
          hb_tensor_t* _result,
          hb_tensor_t* _mat1,
          hb_tensor_t* _mat2) {

    auto mat1 = HBTensor<float, 2>(_mat1);
    auto mat2 = HBTensor<float, 2>(_mat2);
    auto result = HBTensor<float, 2>(_result);

    float* src_ptr = (float*)mat1.data_ptr();
    const uint32_t* src_strides = mat1.get_strides();
    float* src_base = src_ptr + bsg_id * TILE_LOAD_RANGE * src_strides[1];

    register float tmp0;
    register float tmp1;
    register float tmp2;
    register float tmp3;
    register float tmp4;
    register float tmp5;
    register float tmp6;
    register float tmp7;
    register float tmp8;
    register float tmp9;
    register float tmp10;
    register float tmp11;
    register float tmp12;
    register float tmp13;
    register float tmp14;
    register float tmp15;
    register float tmp16;
    register float tmp17;
    register float tmp18;
    register float tmp19;
    register float tmp20;
    register float tmp21;
    register float tmp22;
    register float tmp23;
    register float tmp24;
    register float tmp25;
    register float tmp26;
    register float tmp27;
    register float tmp28;
    register float tmp29;
    register float tmp30;
    register float tmp31;

    float buffer[TILE_LOAD_RANGE];

    for (size_t iter = 1; iter < LOAD_ITER+1; iter++) {
      g_barrier.sync();
      bsg_cuda_print_stat_start(iter);

      // iter 1 -- cold icache cold data
      // iter 2 -- warm icache warm data
      // iter 3 -- warm icache cold data
      if (iter == 3) {
        src_base += src_strides[0];
      }

      float* src_offset = src_base;
      float* buf_offset = buffer;

      for (size_t idx = 0; idx < TILE_LOAD_RANGE/32; idx++) {

        // run ahead 32
        /*
        asm volatile("flw %0,   0(%1)" :  "=f"(tmp0)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   4(%1)" :  "=f"(tmp1)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   8(%1)" :  "=f"(tmp2)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  12(%1)" :  "=f"(tmp3)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  16(%1)" :  "=f"(tmp4)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  20(%1)" :  "=f"(tmp5)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  24(%1)" :  "=f"(tmp6)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  28(%1)" :  "=f"(tmp7)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  32(%1)" :  "=f"(tmp8)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  36(%1)" :  "=f"(tmp9)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  40(%1)" :  "=f"(tmp10) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  44(%1)" :  "=f"(tmp11) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  48(%1)" :  "=f"(tmp12) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  52(%1)" :  "=f"(tmp13) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  56(%1)" :  "=f"(tmp14) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  60(%1)" :  "=f"(tmp15) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  64(%1)" :  "=f"(tmp16) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  68(%1)" :  "=f"(tmp17) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  72(%1)" :  "=f"(tmp18) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  76(%1)" :  "=f"(tmp19) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  80(%1)" :  "=f"(tmp20) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  84(%1)" :  "=f"(tmp21) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  88(%1)" :  "=f"(tmp22) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  92(%1)" :  "=f"(tmp23) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  96(%1)" :  "=f"(tmp24) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 100(%1)" :  "=f"(tmp25) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 104(%1)" :  "=f"(tmp26) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 108(%1)" :  "=f"(tmp27) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 112(%1)" :  "=f"(tmp28) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 116(%1)" :  "=f"(tmp29) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 120(%1)" :  "=f"(tmp30) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 124(%1)" :  "=f"(tmp31) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,   0(%1)" ::  "f"(tmp0),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   4(%1)" ::  "f"(tmp1),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   8(%1)" ::  "f"(tmp2),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  12(%1)" ::  "f"(tmp3),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  16(%1)" ::  "f"(tmp4),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  20(%1)" ::  "f"(tmp5),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  24(%1)" ::  "f"(tmp6),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  28(%1)" ::  "f"(tmp7),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  32(%1)" ::  "f"(tmp8),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  36(%1)" ::  "f"(tmp9),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  40(%1)" ::  "f"(tmp10),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  44(%1)" ::  "f"(tmp11),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  48(%1)" ::  "f"(tmp12),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  52(%1)" ::  "f"(tmp13),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  56(%1)" ::  "f"(tmp14),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  60(%1)" ::  "f"(tmp15),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  64(%1)" ::  "f"(tmp16),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  68(%1)" ::  "f"(tmp17),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  72(%1)" ::  "f"(tmp18),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  76(%1)" ::  "f"(tmp19),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  80(%1)" ::  "f"(tmp20),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  84(%1)" ::  "f"(tmp21),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  88(%1)" ::  "f"(tmp22),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  92(%1)" ::  "f"(tmp23),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  96(%1)" ::  "f"(tmp24),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 100(%1)" ::  "f"(tmp25),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 104(%1)" ::  "f"(tmp26),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 108(%1)" ::  "f"(tmp27),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 112(%1)" ::  "f"(tmp28),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 116(%1)" ::  "f"(tmp29),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 120(%1)" ::  "f"(tmp30),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 124(%1)" ::  "f"(tmp31),  "r"(buf_offset) : "memory");
        */

        // run ahead 16
        /*
        asm volatile("flw %0,   0(%1)" :  "=f"(tmp0)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   4(%1)" :  "=f"(tmp1)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   8(%1)" :  "=f"(tmp2)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  12(%1)" :  "=f"(tmp3)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  16(%1)" :  "=f"(tmp4)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  20(%1)" :  "=f"(tmp5)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  24(%1)" :  "=f"(tmp6)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  28(%1)" :  "=f"(tmp7)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  32(%1)" :  "=f"(tmp8)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  36(%1)" :  "=f"(tmp9)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  40(%1)" :  "=f"(tmp10) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  44(%1)" :  "=f"(tmp11) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  48(%1)" :  "=f"(tmp12) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  52(%1)" :  "=f"(tmp13) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  56(%1)" :  "=f"(tmp14) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  60(%1)" :  "=f"(tmp15) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,   0(%1)" ::  "f"(tmp0),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   4(%1)" ::  "f"(tmp1),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   8(%1)" ::  "f"(tmp2),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  12(%1)" ::  "f"(tmp3),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  16(%1)" ::  "f"(tmp4),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  20(%1)" ::  "f"(tmp5),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  24(%1)" ::  "f"(tmp6),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  28(%1)" ::  "f"(tmp7),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  32(%1)" ::  "f"(tmp8),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  36(%1)" ::  "f"(tmp9),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  40(%1)" ::  "f"(tmp10),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  44(%1)" ::  "f"(tmp11),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  48(%1)" ::  "f"(tmp12),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  52(%1)" ::  "f"(tmp13),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  56(%1)" ::  "f"(tmp14),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  60(%1)" ::  "f"(tmp15),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  64(%1)" :  "=f"(tmp16) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  68(%1)" :  "=f"(tmp17) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  72(%1)" :  "=f"(tmp18) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  76(%1)" :  "=f"(tmp19) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  80(%1)" :  "=f"(tmp20) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  84(%1)" :  "=f"(tmp21) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  88(%1)" :  "=f"(tmp22) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  92(%1)" :  "=f"(tmp23) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  96(%1)" :  "=f"(tmp24) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 100(%1)" :  "=f"(tmp25) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 104(%1)" :  "=f"(tmp26) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 108(%1)" :  "=f"(tmp27) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 112(%1)" :  "=f"(tmp28) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 116(%1)" :  "=f"(tmp29) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 120(%1)" :  "=f"(tmp30) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 124(%1)" :  "=f"(tmp31) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  64(%1)" ::  "f"(tmp16),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  68(%1)" ::  "f"(tmp17),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  72(%1)" ::  "f"(tmp18),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  76(%1)" ::  "f"(tmp19),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  80(%1)" ::  "f"(tmp20),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  84(%1)" ::  "f"(tmp21),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  88(%1)" ::  "f"(tmp22),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  92(%1)" ::  "f"(tmp23),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  96(%1)" ::  "f"(tmp24),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 100(%1)" ::  "f"(tmp25),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 104(%1)" ::  "f"(tmp26),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 108(%1)" ::  "f"(tmp27),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 112(%1)" ::  "f"(tmp28),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 116(%1)" ::  "f"(tmp29),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 120(%1)" ::  "f"(tmp30),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 124(%1)" ::  "f"(tmp31),  "r"(buf_offset) : "memory");
        */

        // run ahead 8
        /*
        asm volatile("flw %0,   0(%1)" :  "=f"(tmp0)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   4(%1)" :  "=f"(tmp1)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   8(%1)" :  "=f"(tmp2)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  12(%1)" :  "=f"(tmp3)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  16(%1)" :  "=f"(tmp4)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  20(%1)" :  "=f"(tmp5)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  24(%1)" :  "=f"(tmp6)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  28(%1)" :  "=f"(tmp7)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,   0(%1)" ::  "f"(tmp0),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   4(%1)" ::  "f"(tmp1),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   8(%1)" ::  "f"(tmp2),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  12(%1)" ::  "f"(tmp3),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  16(%1)" ::  "f"(tmp4),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  20(%1)" ::  "f"(tmp5),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  24(%1)" ::  "f"(tmp6),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  28(%1)" ::  "f"(tmp7),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  32(%1)" :  "=f"(tmp8)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  36(%1)" :  "=f"(tmp9)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  40(%1)" :  "=f"(tmp10) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  44(%1)" :  "=f"(tmp11) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  48(%1)" :  "=f"(tmp12) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  52(%1)" :  "=f"(tmp13) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  56(%1)" :  "=f"(tmp14) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  60(%1)" :  "=f"(tmp15) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  32(%1)" ::  "f"(tmp8),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  36(%1)" ::  "f"(tmp9),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  40(%1)" ::  "f"(tmp10),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  44(%1)" ::  "f"(tmp11),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  48(%1)" ::  "f"(tmp12),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  52(%1)" ::  "f"(tmp13),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  56(%1)" ::  "f"(tmp14),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  60(%1)" ::  "f"(tmp15),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  64(%1)" :  "=f"(tmp16) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  68(%1)" :  "=f"(tmp17) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  72(%1)" :  "=f"(tmp18) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  76(%1)" :  "=f"(tmp19) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  80(%1)" :  "=f"(tmp20) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  84(%1)" :  "=f"(tmp21) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  88(%1)" :  "=f"(tmp22) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  92(%1)" :  "=f"(tmp23) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  64(%1)" ::  "f"(tmp16),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  68(%1)" ::  "f"(tmp17),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  72(%1)" ::  "f"(tmp18),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  76(%1)" ::  "f"(tmp19),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  80(%1)" ::  "f"(tmp20),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  84(%1)" ::  "f"(tmp21),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  88(%1)" ::  "f"(tmp22),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  92(%1)" ::  "f"(tmp23),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  96(%1)" :  "=f"(tmp24) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 100(%1)" :  "=f"(tmp25) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 104(%1)" :  "=f"(tmp26) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 108(%1)" :  "=f"(tmp27) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 112(%1)" :  "=f"(tmp28) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 116(%1)" :  "=f"(tmp29) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 120(%1)" :  "=f"(tmp30) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 124(%1)" :  "=f"(tmp31) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  96(%1)" ::  "f"(tmp24),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 100(%1)" ::  "f"(tmp25),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 104(%1)" ::  "f"(tmp26),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 108(%1)" ::  "f"(tmp27),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 112(%1)" ::  "f"(tmp28),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 116(%1)" ::  "f"(tmp29),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 120(%1)" ::  "f"(tmp30),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 124(%1)" ::  "f"(tmp31),  "r"(buf_offset) : "memory");
        */

        // run ahead 4
        /*
        asm volatile("flw %0,   0(%1)" :  "=f"(tmp0)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   4(%1)" :  "=f"(tmp1)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   8(%1)" :  "=f"(tmp2)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  12(%1)" :  "=f"(tmp3)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,   0(%1)" ::  "f"(tmp0),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   4(%1)" ::  "f"(tmp1),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   8(%1)" ::  "f"(tmp2),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  12(%1)" ::  "f"(tmp3),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  16(%1)" :  "=f"(tmp4)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  20(%1)" :  "=f"(tmp5)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  24(%1)" :  "=f"(tmp6)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  28(%1)" :  "=f"(tmp7)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  16(%1)" ::  "f"(tmp4),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  20(%1)" ::  "f"(tmp5),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  24(%1)" ::  "f"(tmp6),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  28(%1)" ::  "f"(tmp7),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  32(%1)" :  "=f"(tmp8)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  36(%1)" :  "=f"(tmp9)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  40(%1)" :  "=f"(tmp10) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  44(%1)" :  "=f"(tmp11) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  32(%1)" ::  "f"(tmp8),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  36(%1)" ::  "f"(tmp9),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  40(%1)" ::  "f"(tmp10),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  44(%1)" ::  "f"(tmp11),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  48(%1)" :  "=f"(tmp12) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  52(%1)" :  "=f"(tmp13) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  56(%1)" :  "=f"(tmp14) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  60(%1)" :  "=f"(tmp15) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  48(%1)" ::  "f"(tmp12),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  52(%1)" ::  "f"(tmp13),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  56(%1)" ::  "f"(tmp14),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  60(%1)" ::  "f"(tmp15),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  64(%1)" :  "=f"(tmp16) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  68(%1)" :  "=f"(tmp17) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  72(%1)" :  "=f"(tmp18) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  76(%1)" :  "=f"(tmp19) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  64(%1)" ::  "f"(tmp16),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  68(%1)" ::  "f"(tmp17),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  72(%1)" ::  "f"(tmp18),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  76(%1)" ::  "f"(tmp19),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  80(%1)" :  "=f"(tmp20) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  84(%1)" :  "=f"(tmp21) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  88(%1)" :  "=f"(tmp22) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  92(%1)" :  "=f"(tmp23) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  80(%1)" ::  "f"(tmp20),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  84(%1)" ::  "f"(tmp21),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  88(%1)" ::  "f"(tmp22),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  92(%1)" ::  "f"(tmp23),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  96(%1)" :  "=f"(tmp24) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 100(%1)" :  "=f"(tmp25) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 104(%1)" :  "=f"(tmp26) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 108(%1)" :  "=f"(tmp27) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  96(%1)" ::  "f"(tmp24),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 100(%1)" ::  "f"(tmp25),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 104(%1)" ::  "f"(tmp26),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 108(%1)" ::  "f"(tmp27),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 112(%1)" :  "=f"(tmp28) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 116(%1)" :  "=f"(tmp29) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 120(%1)" :  "=f"(tmp30) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 124(%1)" :  "=f"(tmp31) : "r"(src_offset) : "memory");

        asm volatile("fsw %0, 112(%1)" ::  "f"(tmp28),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 116(%1)" ::  "f"(tmp29),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 120(%1)" ::  "f"(tmp30),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 124(%1)" ::  "f"(tmp31),  "r"(buf_offset) : "memory");
        */

        // run ahead 2
        /*
        asm volatile("flw %0,   0(%1)" :  "=f"(tmp0)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,   4(%1)" :  "=f"(tmp1)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,   0(%1)" ::  "f"(tmp0),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,   4(%1)" ::  "f"(tmp1),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,   8(%1)" :  "=f"(tmp2)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  12(%1)" :  "=f"(tmp3)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,   8(%1)" ::  "f"(tmp2),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  12(%1)" ::  "f"(tmp3),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  16(%1)" :  "=f"(tmp4)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  20(%1)" :  "=f"(tmp5)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  16(%1)" ::  "f"(tmp4),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  20(%1)" ::  "f"(tmp5),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  24(%1)" :  "=f"(tmp6)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  28(%1)" :  "=f"(tmp7)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  24(%1)" ::  "f"(tmp6),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  28(%1)" ::  "f"(tmp7),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  32(%1)" :  "=f"(tmp8)  : "r"(src_offset) : "memory");
        asm volatile("flw %0,  36(%1)" :  "=f"(tmp9)  : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  32(%1)" ::  "f"(tmp8),   "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  36(%1)" ::  "f"(tmp9),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  40(%1)" :  "=f"(tmp10) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  44(%1)" :  "=f"(tmp11) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  40(%1)" ::  "f"(tmp10),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  44(%1)" ::  "f"(tmp11),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  48(%1)" :  "=f"(tmp12) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  52(%1)" :  "=f"(tmp13) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  48(%1)" ::  "f"(tmp12),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  52(%1)" ::  "f"(tmp13),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  56(%1)" :  "=f"(tmp14) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  60(%1)" :  "=f"(tmp15) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  56(%1)" ::  "f"(tmp14),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  60(%1)" ::  "f"(tmp15),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  64(%1)" :  "=f"(tmp16) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  68(%1)" :  "=f"(tmp17) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  64(%1)" ::  "f"(tmp16),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  68(%1)" ::  "f"(tmp17),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  72(%1)" :  "=f"(tmp18) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  76(%1)" :  "=f"(tmp19) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  72(%1)" ::  "f"(tmp18),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  76(%1)" ::  "f"(tmp19),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  80(%1)" :  "=f"(tmp20) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  84(%1)" :  "=f"(tmp21) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  80(%1)" ::  "f"(tmp20),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  84(%1)" ::  "f"(tmp21),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  88(%1)" :  "=f"(tmp22) : "r"(src_offset) : "memory");
        asm volatile("flw %0,  92(%1)" :  "=f"(tmp23) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  88(%1)" ::  "f"(tmp22),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0,  92(%1)" ::  "f"(tmp23),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  96(%1)" :  "=f"(tmp24) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 100(%1)" :  "=f"(tmp25) : "r"(src_offset) : "memory");

        asm volatile("fsw %0,  96(%1)" ::  "f"(tmp24),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 100(%1)" ::  "f"(tmp25),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 104(%1)" :  "=f"(tmp26) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 108(%1)" :  "=f"(tmp27) : "r"(src_offset) : "memory");

        asm volatile("fsw %0, 104(%1)" ::  "f"(tmp26),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 108(%1)" ::  "f"(tmp27),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 112(%1)" :  "=f"(tmp28) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 116(%1)" :  "=f"(tmp29) : "r"(src_offset) : "memory");

        asm volatile("fsw %0, 112(%1)" ::  "f"(tmp28),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 116(%1)" ::  "f"(tmp29),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 120(%1)" :  "=f"(tmp30) : "r"(src_offset) : "memory");
        asm volatile("flw %0, 124(%1)" :  "=f"(tmp31) : "r"(src_offset) : "memory");

        asm volatile("fsw %0, 120(%1)" ::  "f"(tmp30),  "r"(buf_offset) : "memory");
        asm volatile("fsw %0, 124(%1)" ::  "f"(tmp31),  "r"(buf_offset) : "memory");
        */


        // no run ahead
        asm volatile("flw %0,   0(%1)" :  "=f"(tmp0)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,   0(%1)" ::  "f"(tmp0),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,   4(%1)" :  "=f"(tmp1)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,   4(%1)" ::  "f"(tmp1),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,   8(%1)" :  "=f"(tmp2)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,   8(%1)" ::  "f"(tmp2),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  12(%1)" :  "=f"(tmp3)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  12(%1)" ::  "f"(tmp3),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  16(%1)" :  "=f"(tmp4)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  16(%1)" ::  "f"(tmp4),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  20(%1)" :  "=f"(tmp5)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  20(%1)" ::  "f"(tmp5),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  24(%1)" :  "=f"(tmp6)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  24(%1)" ::  "f"(tmp6),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  28(%1)" :  "=f"(tmp7)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  28(%1)" ::  "f"(tmp7),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  32(%1)" :  "=f"(tmp8)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  32(%1)" ::  "f"(tmp8),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  36(%1)" :  "=f"(tmp9)  : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  36(%1)" ::  "f"(tmp9),   "r"(buf_offset) : "memory");

        asm volatile("flw %0,  40(%1)" :  "=f"(tmp10) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  40(%1)" ::  "f"(tmp10),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  44(%1)" :  "=f"(tmp11) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  44(%1)" ::  "f"(tmp11),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  48(%1)" :  "=f"(tmp12) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  48(%1)" ::  "f"(tmp12),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  52(%1)" :  "=f"(tmp13) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  52(%1)" ::  "f"(tmp13),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  56(%1)" :  "=f"(tmp14) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  56(%1)" ::  "f"(tmp14),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  60(%1)" :  "=f"(tmp15) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  60(%1)" ::  "f"(tmp15),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  64(%1)" :  "=f"(tmp16) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  64(%1)" ::  "f"(tmp16),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  68(%1)" :  "=f"(tmp17) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  68(%1)" ::  "f"(tmp17),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  72(%1)" :  "=f"(tmp18) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  72(%1)" ::  "f"(tmp18),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  76(%1)" :  "=f"(tmp19) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  76(%1)" ::  "f"(tmp19),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  80(%1)" :  "=f"(tmp20) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  80(%1)" ::  "f"(tmp20),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  84(%1)" :  "=f"(tmp21) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  84(%1)" ::  "f"(tmp21),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  88(%1)" :  "=f"(tmp22) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  88(%1)" ::  "f"(tmp22),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  92(%1)" :  "=f"(tmp23) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  92(%1)" ::  "f"(tmp23),  "r"(buf_offset) : "memory");

        asm volatile("flw %0,  96(%1)" :  "=f"(tmp24) : "r"(src_offset) : "memory");
        asm volatile("fsw %0,  96(%1)" ::  "f"(tmp24),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 100(%1)" :  "=f"(tmp25) : "r"(src_offset) : "memory");
        asm volatile("fsw %0, 100(%1)" ::  "f"(tmp25),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 104(%1)" :  "=f"(tmp26) : "r"(src_offset) : "memory");
        asm volatile("fsw %0, 104(%1)" ::  "f"(tmp26),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 108(%1)" :  "=f"(tmp27) : "r"(src_offset) : "memory");
        asm volatile("fsw %0, 108(%1)" ::  "f"(tmp27),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 112(%1)" :  "=f"(tmp28) : "r"(src_offset) : "memory");
        asm volatile("fsw %0, 112(%1)" ::  "f"(tmp28),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 116(%1)" :  "=f"(tmp29) : "r"(src_offset) : "memory");
        asm volatile("fsw %0, 116(%1)" ::  "f"(tmp29),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 120(%1)" :  "=f"(tmp30) : "r"(src_offset) : "memory");
        asm volatile("fsw %0, 120(%1)" ::  "f"(tmp30),  "r"(buf_offset) : "memory");

        asm volatile("flw %0, 124(%1)" :  "=f"(tmp31) : "r"(src_offset) : "memory");
        asm volatile("fsw %0, 124(%1)" ::  "f"(tmp31),  "r"(buf_offset) : "memory");

        src_offset += 32;
        buf_offset += 32;
      }

      bsg_cuda_print_stat_end(iter);

      float sum = 0;
      for (size_t idx = 0; idx < TILE_LOAD_RANGE; idx++) {
        sum += buffer[idx];
      }
      result(bsg_id,iter) = sum;
    }

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_load_mbmark, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

