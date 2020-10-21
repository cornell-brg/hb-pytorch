//====================================================================
// Vector - vector add kernel
// 06/02/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_DIM 2

extern "C" {

  __attribute__ ((noinline))  int tensorlib_lu(
          hb_tensor_t* factorization_p,
          hb_tensor_t* pivots_p) {

    // Convert all low level pointers to Tensor objects
    HBTensor<float> factorization(factorization_p); // LU: m by n matrix (N by N for now)
    HBTensor<int> pivots(pivots_p); // P: m by 1 vector (N by 1 for now)

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {
        int N = factorization.dim(0); // A is N by N
        float curr_magnitude;
        float max_magnitude;
        int idx_max_magnitude;

        // initialize pivots to main diagonal
        for (size_t i = 0; i < pivots.numel(); i++) {
            pivots(i) = int(i + 1);
        }

        for (size_t i = 0; i < N; i++) { // for diagonal index
            idx_max_magnitude = i;
            max_magnitude = 0.0f;
            // find the row with the max magnitude first element
            for (size_t k = i; k < N; k++) {
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
                for (size_t j = 0; j < N; j++) {
                    float temp_val = factorization(i,j);
                    factorization(i,j) = factorization(idx_max_magnitude,j);
                    factorization(idx_max_magnitude,j) = temp_val;
                }

                // record row swap in P
                int temp_pivot_idx = pivots(i);
                pivots(i) = pivots(idx_max_magnitude);
                pivots(idx_max_magnitude) = temp_pivot_idx;
            }

/*
            size_t n_blk = (N - i + BLOCK_DIM - 1) / BLOCK_DIM; // number of blocks
            size_t last_blk_dim = N % BLOCK_DIM == 0 ? BLOCK_DIM : N % BLOCK_DIM;

            // compute row and col
            for (size_t b = 0; b < n_blk; b++) {
                size_t curr_blk_dim = b == n_blk - 1 ? last_blk_dim : BLOCK_DIM;

                // initialize scratchpad store
                float sp_N[curr_blk_dim * curr_blk_dim];
                float sp_W[curr_blk_dim * curr_blk_dim];

                // initialize result store
                float sp_res[curr_blk_dim];
                memset(sp_res, 0, curr_blk_dim);

                // load from DRAM to scratchpad
                for (size_t s = 0; s < size; s++) {
                    size_t r = s / BLOCK_DIM;
                    size_t c = s % BLOCK_DIM;
                    sp_N[s] = factorization(r, c);
                    sp_W[s] = factorization(c, r);
                }

                // compute row and col
                for (size_t s = 0; s < curr_blk_dim; s++) {
                    sp_res_row[s] = 0;
                    sp_res_col[s] = 0;
                    for (size_t j = 0; j < N-i-1; j++) {
                        sp_res_row[s] += sp_W[j] * sp_N[s+1 + j*];
                        sp_res_col[s] += sp_W[s+1] * sp_N[j*BLOCK_DIM];
                    }
                    //sp_res_row[s] = factorization() - sp_res_row[s];

                }

                // write result back
                for (size_t s = 0; s < curr_blk_dim; s++) {
                    size_t offset = i + b * BLOCK_DIM + s;
                    factorization(offset, i) -= sp_res_row[s];
                    factorization(i, offset) = sp_res_col[s] / ;
                }

                // while blocks are loaded, compute next diagonal elm
                // note that the checks here and the extra load from DRAM
                // should probably be optimized away somehow...
                if (b == 0 && i != N-1) factorization(i, i) = factorization(i+1, i+1) - row[0] * col[0];
            }

*/


            // original, non-blocked version
            for (size_t j = i + 1; j < N; j++) { // for each row below this diagonal
                factorization(j, i) /= factorization(i, i); // calculate L below this diagonal
                for (size_t k = i + 1; k < N; k++) { // for each column in this row
                    factorization(j, k) -= factorization(j, i) * factorization(i, k); // calculate what remains after taking out partial sum so far
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
