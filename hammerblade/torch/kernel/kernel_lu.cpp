//====================================================================
// Vector - vector add kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_lu(
          hb_tensor_t* factorization_p,
          hb_tensor_t* pivots_p) {

    // Convert all low level pointers to Tensor objects
    HBTensor<float> factorization(factorization_p); // LU: m by n matrix
    HBTensor<float> pivots(pivots_p); // P: m by 1 vector

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {

        int N = factorization.dim(0); // A is N by N
        float curr_magnitude;
        float max_magnitude;
        int idx_max_magnitude;

        // initialize pivots to main diagonal
        for (int i = 0; i < pivots.numel(); i++) {
            pivots(i) = i + 1;
        }

        for (int i = 0; i < N; i++) { // for diagonal index
            idx_max_magnitude = i;
            max_magnitude = 0.0f;
            // find the row with the max magnitude first element
            for (int k = i; k < N; k++) {
                curr_magnitude = std::abs(factorization(k, i));
                if (curr_magnitude > max_magnitude) {
                    idx_max_magnitude = k;
                    max_magnitude = curr_magnitude;
                }
            }

            // current row is not the max magnitude row
            // needs pivoting
            if (idx_max_magnitude != i) {
                // swap rows of A
                for (int j = 0; j < N; j++) {
                    float temp_val = factorization(i,j);
                    factorization(i,j) = factorization(idx_max_magnitude,j);
                    factorization(idx_max_magnitude,j) = temp_val;
                }

                // record row swap in P
                int temp_pivot_idx = pivots(i);
                pivots(i) = pivots(idx_max_magnitude);
                pivots(idx_max_magnitude) = temp_pivot_idx;
            }

            for (int j = i + 1; j < N; j++) {
                factorization(j, i) /= factorization(i, i);
                for (int k = i + 1; k < N; k++) {
                    factorization(j, k) -= factorization(j, i) * factorization(i, k);
                }

            }

/*
            float lower, upper;
            // compute U
            printf("----compute U\n");
            for (int k = i; k < N; k++) {
                float sum = 0.0f;
                for (int j = 0; j < i; j++) {
                    lower = (i < j) ? 0 :
                            (i == j) ? 1 :
                            factorization(i, j);
                    upper = (j > k) ? 0 : factorization(j, k);
                    sum += lower * upper;
                }
                factorization(i, k) = factorization(i, k) - sum;
            }

            // compute L
            for (int k = i; k < N; k++) {
                float sum = 0;
                for (int j = 0; j < i; j++) {
                    lower = (k < j) ? 0 :
                            (k == j) ? 1 :
                            factorization(k, j);
                    upper = (j > i) ? 0 : factorization(j, i);
                    sum += lower * upper;
                }

                // don't overwrite U
                if (k > i) factorization(k, i) = (factorization(k, i) - sum) / factorization(i, i);
            }
*/

        }

    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_lu, hb_tensor_t*, hb_tensor_t*)

}
