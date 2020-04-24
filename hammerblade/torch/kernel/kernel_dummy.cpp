//====================================================================
// Dummy kernel for internal testing/benchmarking
// 04/22/2020 Kexin Zheng (kz73@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_SIZE 4

extern "C" {

  __attribute__ ((noinline))  int tensorlib_dummy(
          hb_tensor_t* _result,
          hb_tensor_t* _self,
          hb_tensor_t* _other) {
    auto result = HBTensor<float>(_result);
    auto self = HBTensor<float>(_self);
    auto other = HBTensor<float>(_other);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();


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
    // v5, v6, v7: single vector, block load + store
    // v5: BLOCK_SIZE = 2
    // v6: BLOCK_SIZE = 4
    // v7: BLOCK_SIZE = 8
    int num_blks = self.numel() / BLOCK_SIZE;
    for (int i = 0; i < num_blks; i++) {
        float buffer[BLOCK_SIZE];
        // DRAM -> sp
        for (int j = 0; j < BLOCK_SIZE; j++) {
            buffer[j] = self(i * BLOCK_SIZE + j);
        }
        // sp -> DRAM
        for (int j = 0; j < BLOCK_SIZE; j++) {
            result(i * BLOCK_SIZE + j) = buffer[j];
        }
    }
*/


/*
    // v?: 2 vectors, elm by elm load + calc + store
    int size = self.numel();
    float buffer1[size];
    float buffer2[size];
    float buffer3[size];
    // self DRAM -> sp
    for (int i = 0; i < size; i++) {
        buffer1[i] = self(i);
    }
    // other DRAM -> sp
    for (int i = 0; i < size; i++) {
        buffer2[i] = other(i);
    }
    // compute vec1 + vec2
    for (int i = 0; i < size; i++) {
        buffer3[i] = buffer1[i] + buffer2[i];
    }
    // result sp -> DRAM
    for (int i = 0; i < size; i++) {
        result(i) = buffer3[i];
    }
*/

/*
    // v?: 2 vectors, block load + calc + store
    int num_blks = self.numel() / BLOCK_SIZE;
    for (int i = 0; i < num_blks; i++) {
        float buffer1[BLOCK_SIZE];
        float buffer2[BLOCK_SIZE];
        float buffer3[BLOCK_SIZE];
        // self DRAM -> sp
        for (int j = 0; j < BLOCK_SIZE; j++) {
            buffer1[j] = self(i * BLOCK_SIZE + j);
        }
        // other DRAM -> sp
        for (int j = 0; j < BLOCK_SIZE; j++) {
            buffer1[j] = self(i * BLOCK_SIZE + j);
        }
        // compute vec1 + vec2
        for (int j = 0; j < BLOCK_SIZE; j++) {
            buffer3[j] = buffer1[j] + buffer2[j];
        }
        // result sp -> DRAM
        for (int j = 0; j < BLOCK_SIZE; j++) {
            result(i * BLOCK_SIZE + j) = buffer3[j];
        }
    }
*/





    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_dummy, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
