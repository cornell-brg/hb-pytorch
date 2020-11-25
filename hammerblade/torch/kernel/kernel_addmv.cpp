//====================================================================
// addmv kernel
// 09/26/2020 Krithik Ranjan (kr397@cornell.edu)
//====================================================================

#define BLOCK_SIZE 4 
#define VEC_BLOCK_SIZE 16
#include <kernel_common.hpp>

inline void spcpy(float* dest, float* src) {
  for (int i = 0; i < VEC_BLOCK_SIZE; i += 8) {
        register float tmp0 = *(src + 0);
        register float tmp1 = *(src + 1);
        register float tmp2 = *(src + 2);
        register float tmp3 = *(src + 3);
        register float tmp4 = *(src + 4);
        register float tmp5 = *(src + 5);
        register float tmp6 = *(src + 6);
        register float tmp7 = *(src + 7);
        asm volatile("": : :"memory");
        *(dest + 0) = tmp0;
        *(dest + 1) = tmp1;
        *(dest + 2) = tmp2;
        *(dest + 3) = tmp3;
        *(dest + 4) = tmp4;
        *(dest + 5) = tmp5;
        *(dest + 6) = tmp6;
        *(dest + 7) = tmp7;
        src += 8;
        dest += 8;
  }
}

template<int N, typename Func>
struct Unroll {
  inline static void compute( Func lambda );
};

template<int N, typename Func>
inline void Unroll<N, Func>::compute(Func lambda){
  lambda(N);
  Unroll<N-1, Func>::compute( lambda );
}

template<typename Func>
struct Unroll<0, Func> {
  inline static void compute( Func lambda );
};

template<typename Func>
inline void Unroll<0, Func>::compute( Func lambda ){
  lambda(0);
}

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

        // get data pointers
        float* self_p = (float*) self.data_ptr();
        float* mat_p  = (float*) mat.data_ptr();
        float* vec_p  = (float*) vec.data_ptr();
        float* res_p  = (float*) result.data_ptr();

        // Find out the number of blocks of vec
        int vec_size = vec.dim(0);
        int num_blk = ((int) vec_size / BLOCK_SIZE) + 1;
        
        // Find the range of rows this tile operates on
        int num_rows = ((int) mat.dim(0) / (bsg_tiles_X * bsg_tiles_Y));
        int row_st = __bsg_id * num_rows;
        int row_en = row_st + num_rows;
        row_en = row_en > mat.dim(0) ? mat.dim(0) : row_en;

        // Resultant vector block for this tile
        float res_block[BLOCK_SIZE];

        g_barrier.sync();

        //   Start profiling
        bsg_cuda_print_stat_kernel_start();

        // Outer loop for VEC_BLOCK_SIZE increments over vector
        for ( size_t v = 0; v + VEC_BLOCK_SIZE <= vec_size; v += VEC_BLOCK_SIZE ){
            float vec_block[VEC_BLOCK_SIZE];
            
            // Copy over vec block
            spcpy( vec_block, vec_p );

            // Inner loop for BLOCK_SIZE increments over matrix rows
            for ( size_t row = row_st; row < row_en; row++ ) {
                // Initialize result block
                float res = 0.0;
                
                // Block for matrix elements
                float mat_block[VEC_BLOCK_SIZE];
                spcpy( mat_block, mat_p + ( row * vec_size ) + v );

                float msum = 0.0;
                auto mul_lambda = [&]( size_t i ) {
                    msum += mat_block[i] * vec_block[i];
                };
                Unroll<VEC_BLOCK_SIZE-1, decltype(mul_lambda)>::compute( mul_lambda );

                res_p[row] += beta * msum;
            }
        }

        for ( size_t row = row_st; row < row_en; row++ ) {
            res_p[row] = alpha * self_p[row] + res_p[row];
        }

        /*
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
        */
        
        //   End profiling
        bsg_cuda_print_stat_kernel_end();

        g_barrier.sync();
        return 0;
    }

    HB_EMUL_REG_KERNEL(tensorlib_addmv, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*, float*)
}