//====================================================================
// Vector - vector add kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_lu(
          hb_tensor_t* result_p,
          hb_tensor_t* pivots_p,
          hb_tensor_t* infos_p,
          hb_tensor_t* self_p,
          bool pivot,
          bool get_infos) {

    // Convert all low level pointers to Tensor objects
    HBTensor<float> result(result_p); // LU: m by n matrix
    HBTensor<float> pivots(pivots_p); // P: m by 1 vector
    HBTensor<float> infos(infos_p);  // * by 1 vector
    HBTensor<float> self(self_p);    // A: m by n matrix

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {

        int N = self.dim(0); // A is N by N
        float curr_magnitude;
        float max_magnitude;
        int idx_max_magnitude;

        // initialize pivots to main diagonal
        for (int i = 0; i < pivots.numel(); i++) {
            pivots(i) = i + 1;
        }

        for (int i = 0; i < N; i++) { // for diagonal index
            if (pivot) { // if pivoting should be done
                idx_max_magnitude = i;
                max_magnitude = 0.0f;
                // find the row with the max magnitude first element
                for (int k = i; k < N; k++) {
                    curr_magnitude = std::abs(self(k, i));
                    if (curr_magnitude > max_magnitude) {
                        idx_max_magnitude = k;
                        max_magnitude = curr_magnitude;
                    }
                }

                // current row is not the max magnitude row
                // needs pivoting
                if (idx_max_magnitude != i) {
                    // swap rows of A
                    auto temp_row_ptr = self(i);
                    self(i) = self(idx_max_magnitude);
                    self(idx_max_magnitude) = temp_row_ptr;

                    // record row swap in P
                    int temp_pivot_idx = pivots(i);
                    pivots(i) = pivots(idx_max_magnitude);
                    pivots(idx_max_magnitude) = temp_pivot_idx;
                }
            }

            float lower, upper;

            // compute U
            for (int k = i; k < N; k++) {
                float sum = 0;
                for (int j = 0; j < i; j++) {
                    lower = (i < j) ? 0 :
                            (i == j) ? 1 :
                            result(i, j);
                    upper = (j > k) ? 0 : result(j, k);
                    sum += lower * upper;
                }
                result(i, k) = self(i, k) - sum;
            }

            // compute L
            for (int k = i; k < N; k++) {
                float sum = 0;
                for (int j = 0; j < i; j++) {
                    lower = (k < j) ? 0 :
                            (k == j) ? 1 :
                            result(k, j);
                    upper = (j > i) ? 0 : result(j, i);
                    sum += lower * upper;
                }
                result(k, i) = (self(k, i) - sum) / result(i, i);
            }
        }
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_lu, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, bool, bool)

}
