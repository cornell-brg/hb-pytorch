//====================================================================
// Cholesky kernel
// 11/18/2020 Kexin Zheng (kz73@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#define BLOCK_DIM 8

extern "C" {

    void print_tensor(HBTensor<float>* t) {
        printf("[bsg_id %d\n", __bsg_id);
        int N = (*t).dim(0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f ", (*t)(i, j));
            }
            printf("\n");
        }
        printf("]\n");
    }

    void print_array(float* b, int nrow, int ncol) {
        printf("[bsg_id %d\n", __bsg_id);
        for (int x = 0; x < nrow; x++) {
            for (int y = 0; y < ncol; y++) {
                printf("%f ", b[x*ncol + y]);
            }
            printf("\n");
        }
        printf("]\n");
    }

    // Compute the Cholesky for the given block A and store it in L
    // N is the size of the block (block is square)
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

    // Triangular solve using forwards substitution
    // B = X * L^T
    // L^T(k,j) = L(j,k)
    // X(i,j) = ( B(i,j) - sum_from_k=1_to_j-1 (X(i,k) * L(j,k)) ) / L(j,j)
    // B is the A blocks under the diagonal block, (ncol X ncol)
    // X is the L blocks under the diagonal block, (nrow X ncol)
    // L is chol(diagonal block),                  (ncol X ncol)
    // This function is destructive, i.e. B is overwritten with X
    void block_triangular_solve(float* B, float* L, int nrow, int ncol) {
        for (int i = 0; i < nrow; i++) { // for each row of X
            for (int j = 0; j < ncol; j++) { // for each col in this row
                for (int k = 0; k < j; k++) {
                    // B(i, k) is already calculated since k < j
                    B[i * ncol + j] -= B[i * ncol + k] * L[j * ncol + k];
                }
                B[i * ncol + j] /= L[j * ncol + j];
            }
        }
    }

    // Matrix multiplication, A = A - B * C
    void block_matmul_sub(float* A, float* B, float* C, int nrow, int ncol, int middim) {
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                float sum = 0.0;
                for (int k = 0; k < middim; k++) {
                    sum += B[i * middim + k] * C[k * ncol + j];
                }
                A[i * ncol + j] -= sum;
            }
        }
    }

    // Copies block from DRAM to scratch pad
    // Block is at upper-left elm (y,x) and of size nrow X ncol
    // Row-major storage
    void copy_block_to_sp(float* dst, HBTensor<float>* src, int y, int x, int nrow, int ncol) {
        int idx = 0;
        for (int i = y; i < y + nrow; i++) {
            for (int j = x; j < x + ncol; j++) {
                dst[idx] = (*src)(i, j);
                idx++;
            }
        }
    }

    // Copies block from scratch pad to DRAM
    // Block is at upper-left elm (y,x) and of size nrow X ncol
    // Row-major storage
    void copy_block_to_dram(HBTensor<float>* dst, float* src, int y, int x, int nrow, int ncol) {
        int idx = 0;
        for (int i = y; i < y + nrow; i++) {
            for (int j = x; j < x + ncol; j++) {
                (*dst)(i, j) = src[idx];
                idx++;
            }
        }
    }

    // Set the block at (y,x) with size nrow X ncol in dst to zero
    // Used for the upper triangular portion of L (all zeros)
    void set_block_to_zero(HBTensor<float>* dst, int y, int x, int nrow, int ncol) {
        for (int i = y; i < y + nrow; i++) {
            for (int j = x; j < x + ncol; j++) {
                (*dst)(i, j) = 0.0;
            }
        }
    }

    // Copies the transpose of the block from DRAM to scratch pad to the 1D array
    // Block is at upper-left elm (y,x) and of size nrow X ncol
    void copy_block_transpose_to_sp(float* dst, HBTensor<float>* src, int y, int x, int nrow, int ncol) {
        int idx = 0;
        for (int i = x; i < x + ncol; i++) {
            for (int j = y; j < y + nrow; j++) {
                dst[idx] = (*src)(j, i);
                idx++;
            }
        }
    }

  __attribute__ ((noinline))  int tensorlib_cholesky(
          hb_tensor_t* A_p) {

    // Convert all low level pointers to Tensor objects
    HBTensor<float> A(A_p);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    const int N = A.dim(0); // A is N by N
    const int ntiles = bsg_tiles_X * bsg_tiles_Y;
    const int LAST_BLOCK_DIM = N % BLOCK_DIM == 0 ? BLOCK_DIM : N % BLOCK_DIM; // Dimension of the last diagonal block
    const int nblocks = N / BLOCK_DIM + (N % BLOCK_DIM != 0); // Number of blocks per side

    for (int i = 0; i < nblocks; i++) { // Diagonal index of block
        const int di = i * BLOCK_DIM; // Actual index of upper-left corner of block
        // ne is how many elements are below this diagonal block, also the size of the trailing submatrix
        // If ne is zero or negative, this is the last block
        const int ne = N - di - BLOCK_DIM;
        int curr_blk_dim = (ne > 0) ? BLOCK_DIM : LAST_BLOCK_DIM;

        // Serial Cholesky of diagonal block
        float diag_data[curr_blk_dim * curr_blk_dim];
        float diag_result[curr_blk_dim * curr_blk_dim] = {0};
        copy_block_to_sp(diag_data, &A, di, di, curr_blk_dim, curr_blk_dim);
        g_barrier.sync(); // Make sure everyone got the correct data
        block_cholesky(diag_data, diag_result, curr_blk_dim);

        // Only copy diag block back if I am tile 0
        if (__bsg_id == 0) {
            copy_block_to_dram(&A, diag_result, di, di, curr_blk_dim, curr_blk_dim);
        }

        if (ne <= 0) break; // This is the last diag block, no need for more calculations

        // Parallel triangular solve for all blocks under this diagonal block
        // Each tile is in charge of some 1 X BLOCK_DIM strips of elms
        // Work is evenly divided such that 
        // the first (ntiles - ne%ntiles) tiles handle strip size (ne/ntiles) ("normal chunks")
        // and the rest handle strip size (ne/ntiles)+1 ("big chunks")
        // Each tile processes its elms in blocks
        const int cutoff = ntiles - ne % ntiles;
        const int normal_nstrips = ne / ntiles;
        const int nstrips = (normal_nstrips < 1) ? (__bsg_id < ne) : // Each relevant tile gets one strip
                            (ne % ntiles != 0 && __bsg_id >= cutoff) ? normal_nstrips + 1 :
                             normal_nstrips;
        const int normal_nblocks_per_tile = nstrips / BLOCK_DIM;
        const int nblocks_per_tile = (normal_nblocks_per_tile < 1) ? (nstrips > 0) : 
                                     (nstrips % BLOCK_DIM != 0) ? normal_nblocks_per_tile + 1 :
                                      normal_nblocks_per_tile;

        int x = di; // Since we are computing L, x will always be the same
        int y = 0;
        for (int j = 0; j < nblocks_per_tile; j++) { // Compute L for each block handled by this tile
            // y dimension of this block, x dim is always BLOCK_DIM
            int y_dim = (j == (nblocks_per_tile - 1) && (nstrips % BLOCK_DIM) != 0) ? (nstrips % BLOCK_DIM) : BLOCK_DIM;
            // y index is "diag blocks above me" + "normal chunks before me" + "big chunks before me" + my block offset
            y = (N-ne) + j* BLOCK_DIM + ((normal_nstrips < 1) ? __bsg_id :          // Not enough work for tiles
                                         (__bsg_id < cutoff) ? nstrips * __bsg_id : // I handle normal chunks
                                         (nstrips-1) * cutoff + nstrips * (__bsg_id-cutoff)); // I handle big chunks

            // Triangular solve
            float block_result[y_dim * BLOCK_DIM];
            copy_block_to_sp(block_result, &A, y, x, y_dim, BLOCK_DIM);
            block_triangular_solve(block_result, diag_result, y_dim, BLOCK_DIM);

            // Write back to DRAM
            copy_block_to_dram(&A, block_result, y, x, y_dim, BLOCK_DIM);
            set_block_to_zero(&A, x, y, BLOCK_DIM, y_dim);
        }

        // Make sure all tiles have finished triangular solve before moving on
        g_barrier.sync();

        // Parallel Schur complement update of trailing submatrix
        // Evenly divide the work as before
        // Each tile is in charge of some 1 X ne strips of elms
        // Each tile processes its elms in blocks
        const int nblocks_per_row = ne / BLOCK_DIM + (ne % BLOCK_DIM != 0);
        for (int j = 0; j < nblocks_per_tile; j++) {
            int y_dim = (j == (nblocks_per_tile - 1) && (nstrips % BLOCK_DIM) != 0) ? (nstrips % BLOCK_DIM) : BLOCK_DIM;
            // y index is "diag blocks above me" + "normal chunks before me" + "big chunks before me" + my block offset
            y = (N-ne) + j* BLOCK_DIM + ((normal_nstrips < 1) ? __bsg_id :          // Not enough work for tiles
                                         (__bsg_id < cutoff) ? nstrips * __bsg_id : // I handle normal chunks
                                         (nstrips-1) * cutoff + nstrips * (__bsg_id-cutoff)); // I handle big chunks

            for (int k = 0; k < nblocks_per_row; k++) {
                int x_dim = (k == (nblocks_per_row - 1) && (ne % BLOCK_DIM) != 0) ? ne % BLOCK_DIM : BLOCK_DIM;
                x = di + BLOCK_DIM + k * BLOCK_DIM; // Trailing submatrix starts at di + BLOCK_DIM

                // Copy value of L (row matrix) into scratchpad
                float L_data[y_dim * BLOCK_DIM]; // y_dim X BLOCK_DIM
                copy_block_to_sp(L_data, &A, y, di, y_dim, BLOCK_DIM);

                // Copy value of LT (column matrix) into scratchpad
                float LT_data[BLOCK_DIM * x_dim]; // BLOCK_DIM X x_dim
                copy_block_transpose_to_sp(LT_data, &A, x, di, x_dim, BLOCK_DIM);

                // Copy value of A into scratchpad
                float A_data[y_dim * x_dim];
                copy_block_to_sp(A_data, &A, y, x, y_dim, x_dim);

                // A = A - L*L^T (subtract L*L^T from trailing submatrix)
                block_matmul_sub(A_data, L_data, LT_data, y_dim, x_dim, BLOCK_DIM);

                // Copy back updated value to DRAM
                copy_block_to_dram(&A, A_data, y, x, y_dim, x_dim);
            }
        }

        // Make sure all tiles have finished updating trailing submatrix before moving on
        g_barrier.sync();
    }

    // Use a single tile only
    if (__bsg_id == 0) {

/*
        float data[N * N] = {0};
        float result[N * N] = {0};
        copy_block_to_sp(data, &A, 0, 0, N);
        block_cholesky(data, result, N);
        copy_block_to_dram(&L, result, 0, 0, N);
*/

/*
        for (size_t i = 0; i < N; i++) { // for diagonal index
            for (size_t j = 0; j < i+1; j++) { // for each elm in the row this diag belongs to
                float sum = 0.0;
                for (size_t k = 0; k < j; k++) { // for each elm e to the left of (i,j)
                    sum += L(i, k) * L(j, k); // sum e times elm in jth row directly above e
                }
                L(i, j) = (i == j) ? sqrt(A(i, i) - sum) : ((A(i, j) - sum) / L(j, j));
            }

        }
*/

    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_cholesky, hb_tensor_t*)

}
