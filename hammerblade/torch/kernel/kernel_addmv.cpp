//====================================================================
// addmv kernel
// 09/26/2020 Krithik Ranjan (kr397@cornell.edu)
//====================================================================

#define BLOCK_SIZE 4 
#include <kernel_common.hpp>

extern "C" {

    __attribute__ ((noinline)) int tensorlib_addmv(
            hb_tensor_t* _result, 
            hb_tensor_t* _self, 
            hb_tensor_t* _mat,
            hb_tensor_t* _vec, 
            float* _beta, 
            float* _alpha
    ){
        auto self   = HBTensor<float>(_self);
        auto mat    = HBTensor<float, 2>(_mat);
        auto vec    = HBTensor<float>(_vec);
        auto result = HBTensor<float>(_result);
        float beta  = *_beta;
        float alpha = *_alpha;

        // Find out the number of blocks of vec
        int vec_size = vec.dim(0);
        int num_blk = ((int) vec_size / BLOCK_SIZE) + 1;
        
        // Find the range of rows this tile operates on
        int num_rows = ((int) mat.dim(0) / (bsg_tiles_X * bsg_tiles_Y)) + 1;
        int row_st = __bsg_id * num_rows;
        int row_en = row_st + num_rows;
        row_en = row_en > mat.dim(0) ? mat.dim(0) : row_en;

        // Resultant vector block for this tile
        float res[BLOCK_SIZE];

        // Outer loop for BLOCK_SIZE increments from row_st
        for ( ; row_st + BLOCK_SIZE < row_en; row_st += BLOCK_SIZE ) {
            // Initialize Resultant block
            for ( int i = 0; i < BLOCK_SIZE; i++ )
                res[i] = 0.0;
            
            // Iterate over blocks in the operand vector
            float blk[BLOCK_SIZE];
            int v = 0;
            for (v = 0; v + BLOCK_SIZE < vec_size; v += BLOCK_SIZE ) {
                // Copy vector elements
                for ( int i = 0; i < BLOCK_SIZE; i++ ) {
                    blk[i] = vec(i + v);
                }

                // Iterate over matrix rows to update res
                // i : iterates from row_st to row_st + BLOCK_SIZE
                // j : iterates from v to v + BLOCK_SIZE
                for ( int i = 0; i < BLOCK_SIZE; i++ ) {
                    for ( int j = 0; j < BLOCK_SIZE; j++ ) {
                        res[i] += mat(row_st + i, v + j) * blk[j];
                    }
                }
            }
            // Account for remaining elements in the vector 
            if (v < vec_size) {
                for ( int i = v; i < vec_size; i++ ) {
                    blk[i - v] = vec(i);
                }

                for (int i = 0; i < BLOCK_SIZE; i++ ) {
                    for ( int j = v; j < vec_size; j++ ) {
                        res[i] += mat(row_st + i, j) * blk[j - v];
                    }
                } 
            }

            // Copy over result to result vector
            for ( int i = 0; i < BLOCK_SIZE; i++ ) {
                result(row_st + i) = alpha * self(row_st + i) + beta * res[i];
                //result(row_st + i) = 0;
            }
        }
        if ( row_st < row_en ) {
            // Remaining elements outside the block
            for ( int i = 0; i < BLOCK_SIZE; i++ )
                res[i] = 0.0;

            // Iterate over blocks in the operand vector
            float blk[BLOCK_SIZE];
            int v = 0;
            for (v = 0; v + BLOCK_SIZE < vec_size; v += BLOCK_SIZE ) {
                // Copy vector elements
                for ( int i = 0; i < BLOCK_SIZE; i++ ) {
                    blk[i] = vec(i + v);
                }

                // Iterate over matrix rows to update res
                // i : iterates from row_st to row_st + BLOCK_SIZE
                // j : iterates from v to v + BLOCK_SIZE
                for ( int i = 0; i < row_en - row_st; i++ ) {
                    for ( int j = 0; j < BLOCK_SIZE; j++ ) {
                        res[i] += mat(row_st + i, v + j) * blk[j];
                    }
                }
            }
            // Account for remaining elements in the vector 
            if (v < vec_size) {
                for ( int i = v; i < vec_size; i++ ) {
                    blk[i - v] = vec(i);
                }

                for (int i = 0; i < row_en - row_st; i++ ) {
                    for ( int j = v; j < vec_size; j++ ) {
                        res[i] += mat(row_st + i, j) * blk[j - v];
                    }
                } 
            }

            // Copy over resultant elements
            for ( int i = 0; row_st < row_en; row_st++, i++ ) {
                result(row_st) = alpha * self(row_st) + beta * res[i];
                //result(row_st) = res[i];
            }
        }
        
        //   End profiling
        bsg_cuda_print_stat_kernel_end();

        g_barrier.sync();
        return 0;
    }

    HB_EMUL_REG_KERNEL(tensorlib_addmv, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*, float*)
}