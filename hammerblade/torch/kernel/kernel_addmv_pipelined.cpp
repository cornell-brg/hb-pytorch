//====================================================================
// addmv kernel
// 11/12/2020 Krithik Ranjan (kr397@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <kernel_circular_buffer.hpp>
#define BLOCK_DIM 8

inline void spcpy(float* dest, float* src) {
  for (int i = 0; i < BLOCK_DIM; i += 8) {
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

    __attribute__ ((noinline)) int tensorlib_addmv_pipelined(
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

        // Find the size of the vector
        int vec_size = vec.dim(0);
        
        // Find the range of rows and columns this tile operates on
        int num_cols  = ((int) mat.dim(1)) / ( bsg_tiles_X - 1 );
        int num_rows  = ((int) mat.dim(0)) / ( bsg_tiles_Y - 1 );

        int row_start = (bsg_y - 1) * num_rows;
        int row_end   = row_start + num_rows;
        row_end       = row_end > mat.dim(0) ? mat.dim(0) : row_end;
        int col_start = bsg_x * num_cols;
        int col_end   = col_start + num_cols;
        col_end       = col_end > mat.dim(1) ? mat.dim(1) : col_end;

        // Config
        // 0 -- idle
        // 1 -- Start accumulation
        // 2 -- Read vector elements
        // 3 -- compute
        // 4 -- vec end + compute
        // 5 -- accumulate end + right result
        // 6 -- vec end + accumulate start + compute

        char systolic_config[8][16] = {
        {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0},
        {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
        {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
        {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
        {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
        {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
        {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5},
        {6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5},
        };

        // Activate config
        char (&mc_config)[8][16] = systolic_config;
        char tile_config = mc_config[bsg_y][bsg_x];

        bsg_print_hexadecimal(0xFFFF0000 + (int)tile_config);

        // buffers -- with double buffering
        float acc_block[BLOCK_DIM];
        float mat_block[BLOCK_DIM];
        float* sp_vec;
        float* sp_acc;
        float* sp_vec_remote;
        float* sp_acc_remote;

        CircularBuffer::FIFO<float, BLOCK_DIM, 2> acc_fifo(bsg_y, bsg_x-1, bsg_y, bsg_x+1);
        CircularBuffer::FIFO<float, BLOCK_DIM, 2> vec_fifo(bsg_y-1, bsg_x, bsg_y+1, bsg_x);

        auto compute_acc_start_task = [&] () {
            // Outer loop: num_rows / BLOCK_DIM
            for ( int row = row_start; row < row_end; row += BLOCK_DIM ) {
                // Initialize acc_block
                // for ( int i = 0; i < BLOCK_DIM; i++ ) {
                //     acc_block[i] = 0;
                // }
                auto acc_clr = [&]( size_t i ) {
                    acc_block[i] = 0;
                };
                Unroll<BLOCK_DIM-1, decltype(acc_clr)>::compute( acc_clr );

                // Middle loop: num_cols / BLOCK_DIM
                for ( int c = col_start; c + BLOCK_DIM <= col_end; c += BLOCK_DIM ) {
                    // Get vector elements from c to c + BLOCK_DIM from previous tile
                    sp_vec = vec_fifo.obtain_rd_ptr();

                    // Write vec elements for next tile
                    sp_vec_remote = vec_fifo.obtain_wr_ptr();
                    spcpy(sp_vec_remote, sp_vec);
                    vec_fifo.finish_wr_ptr();

                    for ( int r = row; r < row + BLOCK_DIM; r++ ) {
                        float msum = 0; 

                        // Copy mat[r, c] to mat[r, c+BLOCK_DIM] to sp
                        spcpy(mat_block, mat_p + (r * vec_size) + c);
                        
                        auto mul_lambda = [&]( size_t i ) {
                            msum += mat_block[i] * sp_vec[i];
                        };
                        Unroll<BLOCK_DIM-1, decltype(mul_lambda)>::compute( mul_lambda );
                        // Calculate
                        // for ( int i = 0; i < BLOCK_DIM; i++ ) {
                        //     msum += mat_block[i] * sp_vec[i];
                        // }

                        // Update partial sum
                        acc_block[r - row] += msum;
                    }
                    vec_fifo.finish_rd_ptr();
                }

                // Send partial sum block to next horizontal tile
                sp_acc_remote = acc_fifo.obtain_wr_ptr();
                spcpy(sp_acc_remote, acc_block);
                acc_fifo.finish_wr_ptr();
            }
        };

        auto vec_dma_task = [&] () {
            // Send vector segments as many times as there are row blocks
            for ( int r = 0; r < num_rows; r += BLOCK_DIM ) {
                for ( int c = col_start; c + BLOCK_DIM <= col_end; c += BLOCK_DIM ) {
                    // Read vec into sp array
                    spcpy(mat_block, vec_p + c);
                    
                    // Copy vector elements from dram to sp of next tile
                    // Maybe useful to copy elements to self scratchpad first?
                    sp_vec_remote = vec_fifo.obtain_wr_ptr();
                    spcpy(sp_vec_remote, mat_block);
                    vec_fifo.finish_wr_ptr();
                }
            }
        };

        auto compute_acc_task = [&] () {
            // Outer loop: num_rows / BLOCK_DIM
            for ( int row = row_start; row < row_end; row += BLOCK_DIM ) {
                // Initialize acc_block
                // for ( int i = 0; i < BLOCK_DIM; i++ ) {
                //     acc_block[i] = 0;
                // }
                auto acc_clr = [&]( size_t i ) {
                    acc_block[i] = 0;
                };
                Unroll<BLOCK_DIM-1, decltype(acc_clr)>::compute( acc_clr );

                // Middle loop: num_cols / BLOCK_DIM
                for ( int c = col_start; c + BLOCK_DIM <= col_end; c += BLOCK_DIM ) {
                    // Get vector elements from c to c + BLOCK_DIM from previous tile
                    sp_vec = vec_fifo.obtain_rd_ptr();

                    // Write vec elements for next tile
                    sp_vec_remote = vec_fifo.obtain_wr_ptr();
                    spcpy(sp_vec_remote, sp_vec);
                    vec_fifo.finish_wr_ptr();

                    for ( int r = row; r < row + BLOCK_DIM; r++ ) {
                        float msum = 0; 

                        // Copy mat[r, c] to mat[r, c+BLOCK_DIM] to sp
                        spcpy(mat_block, mat_p + (r * vec_size) + c);
                        
                        // Calculate
                        auto mul_lambda = [&]( size_t i ) {
                            msum += mat_block[i] * sp_vec[i];
                        };
                        Unroll<BLOCK_DIM-1, decltype(mul_lambda)>::compute( mul_lambda );
                        // for ( int i = 0; i < BLOCK_DIM; i++ ) {
                        //     msum += mat_block[i] * sp_vec[i];
                        // }

                        // Update partial sum
                        acc_block[r - row] += msum;
                    }
                    vec_fifo.finish_rd_ptr();
                }
                // Read partial sum from previous tile
                sp_acc = acc_fifo.obtain_rd_ptr();
                // Sum of current accumulate and previous accumulate
                // for ( int i = 0; i < BLOCK_DIM; i++ ){
                //     acc_block[i] += sp_acc[i];
                // }
                auto acc_lambda = [&]( size_t i ) {
                    acc_block[i] += sp_acc[i];
                };
                Unroll<BLOCK_DIM-1, decltype(acc_lambda)>::compute( acc_lambda );

                // Finish reading partial sum
                acc_fifo.finish_rd_ptr();

                // Send partial sum block to next horizontal tile
                sp_acc_remote = acc_fifo.obtain_wr_ptr();
                spcpy(sp_acc_remote, acc_block);
                acc_fifo.finish_wr_ptr();
            }
        };

        auto compute_vec_end_task = [&] () {
            // Outer loop: num_rows / BLOCK_DIM
            for ( int row = row_start; row < row_end; row += BLOCK_DIM ) {
                // Initialize acc_block
                // for ( int i = 0; i < BLOCK_DIM; i++ ) {
                //     acc_block[i] = 0;
                // }
                auto acc_clr = [&]( size_t i ) {
                    acc_block[i] = 0;
                };
                Unroll<BLOCK_DIM-1, decltype(acc_clr)>::compute( acc_clr );

                // Middle loop: num_cols / BLOCK_DIM
                for ( int c = col_start; c + BLOCK_DIM <= col_end; c += BLOCK_DIM ) {
                    // Get vector elements from c to c + BLOCK_DIM from previous tile
                    sp_vec = vec_fifo.obtain_rd_ptr();

                    for ( int r = row; r < row + BLOCK_DIM; r++ ) {
                        float msum = 0; 

                        // Copy mat[r, c] to mat[r, c+BLOCK_DIM] to sp
                        spcpy(mat_block, mat_p + (r * vec_size) + c);
                        
                        // Calculate
                        auto mul_lambda = [&]( size_t i ) {
                            msum += mat_block[i] * sp_vec[i];
                        };
                        Unroll<BLOCK_DIM-1, decltype(mul_lambda)>::compute( mul_lambda );
                        // for ( int i = 0; i < BLOCK_DIM; i++ ) {
                        //     msum += mat_block[i] * sp_vec[i];
                        // }

                        // Update partial sum
                        acc_block[r - row] += msum;
                    }
                    vec_fifo.finish_rd_ptr();
                }
                // Read partial sum from previous tile
                sp_acc = acc_fifo.obtain_rd_ptr();
                // Sum of current accumulate and previous accumulate
                // for ( int i = 0; i < BLOCK_DIM; i++ ){
                //     acc_block[i] += sp_acc[i];
                // }
                auto acc_lambda = [&]( size_t i ) {
                    acc_block[i] += sp_acc[i];
                };
                Unroll<BLOCK_DIM-1, decltype(acc_lambda)>::compute( acc_lambda );
                
                acc_fifo.finish_rd_ptr();

                // Send partial sum block to next horizontal tile
                sp_acc_remote = acc_fifo.obtain_wr_ptr();
                spcpy(sp_acc_remote, acc_block);
                acc_fifo.finish_wr_ptr();
            }
        };

        auto compute_vec_acc_task = [&] () {
            // Outer loop: num_rows / BLOCK_DIM
            for ( int row = row_start; row < row_end; row += BLOCK_DIM ) {
                // Initialize acc_block
                //for ( int i = 0; i < BLOCK_DIM; i++ ) {
                //    acc_block[i] = 0;
                //}
                auto acc_clr = [&]( size_t i ) {
                    acc_block[i] = 0;
                };
                Unroll<BLOCK_DIM-1, decltype(acc_clr)>::compute( acc_clr );

                // Middle loop: num_cols / BLOCK_DIM
                for ( int c = col_start; c + BLOCK_DIM <= col_end; c += BLOCK_DIM ) {
                    // Get vector elements from c to c + BLOCK_DIM from previous tile
                    sp_vec = vec_fifo.obtain_rd_ptr();

                    for ( int r = row; r < row + BLOCK_DIM; r++ ) {
                        float msum = 0; 

                        // Copy mat[r, c] to mat[r, c+BLOCK_DIM] to sp
                        spcpy(mat_block, mat_p + (r * vec_size) + c);
                        
                        // Calculate
                        auto mul_lambda = [&]( size_t i ) {
                            msum += mat_block[i] * sp_vec[i];
                        };
                        Unroll<BLOCK_DIM-1, decltype(mul_lambda)>::compute( mul_lambda );
                        // for ( int i = 0; i < BLOCK_DIM; i++ ) {
                        //     msum += mat_block[i] * sp_vec[i];
                        // }

                        // Update partial sum
                        acc_block[r - row] += msum;
                    }
                    vec_fifo.finish_rd_ptr();
                }

                // Send partial sum block to next horizontal tile
                sp_acc_remote = acc_fifo.obtain_wr_ptr();
                spcpy(sp_acc_remote, acc_block);
                acc_fifo.finish_wr_ptr();
            }
        };

        auto compute_result = [&] () {
            for ( int row = row_start; row + BLOCK_DIM <= row_end; row += BLOCK_DIM ) {
                // Get self vector values
                spcpy(acc_block, self_p + row);

                // bsg_print_hexadecimal(0x500B100D);

                // Obtain accumulated values
                sp_acc = acc_fifo.obtain_rd_ptr();

                auto acc_lambda = [&]( size_t i ) {
                    acc_block[i] = ( alpha * sp_acc[i] ) + ( beta * acc_block[i] );
                };
                Unroll<BLOCK_DIM-1, decltype(acc_lambda)>::compute( acc_lambda );
                // for ( int r = 0; r < BLOCK_DIM; r++ ) {
                //     acc_block[r] = ( alpha * sp_acc[r] ) + ( beta * acc_block[r] );
                //     // acc_block[r] = 1;
                //     // res_p[row+r] = ( alpha * sp_acc[r] ) + ( beta * acc_block[r] );
                // }

                // Finish reading
                acc_fifo.finish_rd_ptr();

                // Write result 
                spcpy(res_p + row, acc_block);
            }
        };


        g_barrier.sync();

        bsg_cuda_print_stat_kernel_start();

        // schedule
        switch (tile_config) {
        case 0:
            // nothing
            break;
        case 1:
            // Start accumulation
            compute_acc_start_task();
            break;
        case 2:
            // Obtain vector
            vec_dma_task();
            break;
        case 3:
            // Compute and accumulate from previous tile
            compute_acc_task();
            break;
        case 4:
            // PolyA Col
            compute_vec_end_task();
            break;
        case 5:
            // Compute and write result
            compute_result();
            break;
        case 6:
            // Accumulate start vec end
            compute_vec_acc_task();
            break;
        }


        bsg_cuda_print_stat_kernel_end();

        g_barrier.sync();
        return 0;
    }

  HB_EMUL_REG_KERNEL(tensorlib_systolic, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}