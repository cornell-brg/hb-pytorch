//====================================================================
// Vector - vector add kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_DIM 2

extern "C" {

    void block_cholesky(float* A, float* L, int N) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < i + 1; j++) {
                float sum = 0.0;
                for (size_t k = 0; k < j; k++) {
                    sum += L[i * N + k] * L[j * N + k]; // sum e times elm in jth row directly above e
                }
                L[i * N + j] = (i == j) ? sqrt(A[i * N + i] - sum) : ((A[i * N + j] - sum) / L[j * N + j]);
            }
        }
    }

  __attribute__ ((noinline))  int tensorlib_cholesky(
          hb_tensor_t* A_p,
          hb_tensor_t* L_p) {

    // Convert all low level pointers to Tensor objects
    HBTensor<float> A(A_p); // LL^T (N by N matrix)
    HBTensor<float> L(L_p); // L (N by N matrix)
    //HBTensor<int> pivots(pivots_p); // P: m by 1 vector (N by 1 for now)

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {
        int N = A.dim(0); // A is N by N

        for (size_t i = 0; i < N; i++) { // for diagonal index
            for (size_t j = 0; j < i+1; j++) { // for each elm in the row this diag belongs to
                float sum = 0.0;
                for (size_t k = 0; k < j; k++) { // for each elm e to the left of (i,j)
                    sum += L(i, k) * L(j, k); // sum e times elm in jth row directly above e
                }
                L(i, j) = (i == j) ? sqrt(A(i, i) - sum) : ((A(i, j) - sum) / L(j, j));
            }

        }

//        for (size_t i = 0; i < N; i++) {
//            for (size_t j = 0; j < N; j++) {
//                printf("%f ", L(i,j));
//            }
//            printf("\n");
//        }

    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_cholesky, hb_tensor_t*, hb_tensor_t*)

}
