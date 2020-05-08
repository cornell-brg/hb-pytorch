//====================================================================
// Blocked Accumulative Sum kernel for internal micro-benchmarking
// 04/22/2020 Kexin Zheng (kz73@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_SIZE 4
#define M 4

extern "C" {

  __attribute__ ((noinline))  int tensorlib_blocked_cum_sum(
          hb_tensor_t* _result,
          hb_tensor_t* _self) {
    auto result = HBTensor<float>(_result);
    auto self = HBTensor<float>(_self);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

/*
    // v1: single vector, elm by elm load + store
    int size = self.numel();
    float buffer[size];
    // DRAM -> sp
    for (int i = 0; i < size; i++) {
        buffer[i] = self(i);
    }
    // sp -> DRAM
    for (int i = 0; i < size; i++) {
        result(i) = buffer[i];
    }
*/

/*
    // v2: unrolled by 2 v1: 
    // single vector, elm by elm load + store
    int size = self.numel();
    float buffer[size];
    // DRAM -> sp
    for (int i = 0; i < size; i += 2) {
        buffer[i] = self(i);
        buffer[i+1] = self(i+1);
    }
    // sp -> DRAM
    for (int i = 0; i < size; i += 2) {
        result(i) = buffer[i];
        result(i+1) = buffer[i+1];
    }
*/

/*
    // v3: unrolled by 4 v1: 
    // single vector, elm by elm load + store
    int size = self.numel();
    float buffer[size];
    // DRAM -> sp
    for (int i = 0; i < size; i += 4) {
        buffer[i] = self(i);
        buffer[i+1] = self(i+1);
        buffer[i+2] = self(i+2);
        buffer[i+3] = self(i+3);
    }
    // sp -> DRAM
    for (int i = 0; i < size; i += 4) {
        result(i) = buffer[i];
        result(i+1) = buffer[i+1];
        result(i+2) = buffer[i+2];
        result(i+3) = buffer[i+3];
    }
*/

/*
    // v4: unrolled by 8 v1: 
    // single vector, elm by elm load + store
    int size = self.numel();
    float buffer[size];
    // DRAM -> sp
    for (int i = 0; i < size; i += 8) {
        buffer[i] = self(i);
        buffer[i+1] = self(i+1);
        buffer[i+2] = self(i+2);
        buffer[i+3] = self(i+3);
        buffer[i+4] = self(i+4);
        buffer[i+5] = self(i+5);
        buffer[i+6] = self(i+6);
        buffer[i+7] = self(i+7);
    }
    // sp -> DRAM
    for (int i = 0; i < size; i += 8) {
        result(i) = buffer[i];
        result(i+1) = buffer[i+1];
        result(i+2) = buffer[i+2];
        result(i+3) = buffer[i+3];
        result(i+4) = buffer[i+4];
        result(i+5) = buffer[i+5];
        result(i+6) = buffer[i+6];
        result(i+7) = buffer[i+7];
    }
*/

/*
    // naive accumulative sum
    int num_blks = self.numel() / BLOCK_SIZE;
    for (int i = 0; i < num_blks; i++) { // which block
        int offset = i * BLOCK_SIZE;
        for (int times = 0; times < M; times++) { // how many times to compute
            for (int j = 0; j < BLOCK_SIZE; j++) { // which elm in this blk
                for (int k = 0; k <= j; k++) { // which number to add
                    result(offset + k) += self(offset + k);
                }
            }
        }
    }
*/

    // blocked buffered accumulative sum
    // BLOCK_SIZE = 4
    int num_blks = self.numel() / BLOCK_SIZE;
    for (int i = 0; i < num_blks; i++) {
        int offset = i * BLOCK_SIZE;
        float buffer1[BLOCK_SIZE]; // store elements
        float buffer2[BLOCK_SIZE]; // store computed result
        // self DRAM -> sp
        for (int j = 0; j < BLOCK_SIZE; j++) {
            buffer2[j] = 0;
            buffer1[j] = self(offset + j);
        }
        // compute accumulative sum
        for (int times = 0; times < M; times++) { // how many times to compute
            for (int j = 0; j < BLOCK_SIZE; j++) { // which elm in this blk
                for (int k = 0; k <= j; k++) { // which number to add
                    buffer2[j] += buffer1[k]; 
                }
            }
        }
        // result sp -> DRAM
        for (int j = 0; j < BLOCK_SIZE; j++) {
            result(offset + j) = buffer2[j];
        }
    }





    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_blocked_cum_sum, hb_tensor_t*, hb_tensor_t*)

}
