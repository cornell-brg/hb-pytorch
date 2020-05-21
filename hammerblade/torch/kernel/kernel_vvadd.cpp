//====================================================================
// Vector-vector Add kernel for internal micro-benchmarking
// 04/22/2020 Kexin Zheng (kz73@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_SIZE 4
#define M 16

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vvadd(
          hb_tensor_t* _result,
          hb_tensor_t* _self,
          hb_tensor_t* _other) {
    auto result = HBTensor<float>(_result);
    auto self = HBTensor<float>(_self);
    auto other = HBTensor<float>(_other);
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

/*
    float self_buffer[BLOCK_SIZE];
    //float other_buffer[BLOCK_SIZE];
    float res_buffer[BLOCK_SIZE];
    int num_blks = self.numel() / BLOCK_SIZE;
    for (int i = 0; i < num_blks; i++) {
        int offset = i * BLOCK_SIZE;

        // DRAM -> sp load data
        for (int j = 0; j < BLOCK_SIZE; j++) {
            self_buffer[j] = self(offset + j);
            //other_buffer[j] = other(offset + j);
            res_buffer[j] = 0;
        }

        // compute
        for (int times = 0; times < M; times++) { // take sum this many times
            for (int j = 0; j < BLOCK_SIZE; j++) {
                res_buffer[j] += self_buffer[j]; //+ other_buffer[j];
            }
        }

        // sp -> DRAM store data
        for (int j = 0; j < BLOCK_SIZE; j++) {
            result(offset + j) = res_buffer[j];
        }
    }
*/

    bsg_printf("RESULT %d\n", result.numel());
    for (int i = 0; i < result.numel(); i++) {
        bsg_print_float(result(i));
    }
    bsg_printf("self %d\n", self.numel());
    for (int i = 0; i < self.numel(); i++) {
        bsg_print_float(self(i));
    }
    bsg_printf("other %d\n", other.numel());
    for (int i = 0; i < other.numel(); i++) {
        bsg_print_float(other(i));
    }

    hb_foreach(result, self, other,
        [&](float self, float other) {
          return self + other;
        });


    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_vvadd, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
