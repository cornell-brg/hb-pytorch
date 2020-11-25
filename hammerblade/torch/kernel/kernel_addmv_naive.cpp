//====================================================================
// addmv kernel
// 09/26/2020 Krithik Ranjan (kr397@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

extern "C" {

    __attribute__ ((noinline)) void addmv_helper(
        HBTensor<float> self, 
        HBTensor<float, 2> mat, 
        HBTensor<float> vec, 
        HBTensor<float> result,
        float alpha, 
        float beta, 
        int row_start,
        int row_end,
        int vec_size
    ) {
        // Iterate over row range
        for ( int r = row_start; r < row_end; r += 3 ) {
            float res0 = 0;
            float res1 = 0;
            float res2 = 0;
            //float res3 = 0;
            //float res4 = 0;
            //float res5 = 0;

            for ( int c = 0; c < vec_size; c++ ) {
                float foo = vec(c);

                res0 += foo * mat(r, c);
                res1 += foo * mat(r+1, c);
                res2 += foo * mat(r+2, c);
                //res3 += foo * mat(r+3, c);
                //res4 += foo * mat(r+4, c);
                //res5 += foo * mat(r+5, c);
            }

            // write to the result tensor
            result(r) = alpha * self(r) + beta * res0;
            result(r+1) = alpha * self(r+1) + beta * res1;
            result(r+2) = alpha * self(r+2) + beta * res2;
            //result(r+3) = alpha * self(r+3) + beta * res3;
            //result(r+4) = alpha * self(r+4) + beta * res4;
            //result(r+5) = alpha * self(r+5) + beta * res5;
        }

    }

    __attribute__ ((noinline)) void addmv_helper_p(
        float* self, 
        float* mat, 
        float* vec, 
        float* result,
        float alpha, 
        float beta, 
        int row_start,
        int row_end,
        int vec_size
    ) {
        // Iterate over row range
        for ( int r = row_start; r < row_end; r += 1 ) {
            float res0 = 0;
            float res1 = 0;
            float res2 = 0;
            float res3 = 0;
            float res4 = 0;
            float res5 = 0;

            for ( int c = 0; c < vec_size; c++ ) {
                float foo = vec[c];

                res0 += foo * mat[r * vec_size + c];
                res1 += foo * mat[(r+1) * vec_size + c];
                res2 += foo * mat[(r+2) * vec_size + c];
                res3 += foo * mat[(r+3) * vec_size + c];
                res4 += foo * mat[(r+4) * vec_size + c];
                res5 += foo * mat[(r+5) * vec_size + c];
            }

            // write to the result tensor
            result[r] = alpha * self[r] + beta * res0;
            result[r+1] = alpha * self[r+1] + beta * res1;
            result[r+2] = alpha * self[r+2] + beta * res2;
            result[r+3] = alpha * self[r+3] + beta * res3;
            result[r+4] = alpha * self[r+4] + beta * res4;
            result[r+5] = alpha * self[r+5] + beta * res5;
        }
    }

    __attribute__ ((noinline)) int tensorlib_addmv_naive(
            hb_tensor_t* _result, 
            hb_tensor_t* _self, 
            hb_tensor_t* _mat,
            hb_tensor_t* _vec, 
            hb_tensor_t* _acc,
            float* _beta, 
            float* _alpha
    ){
        auto self   = HBTensor<float>(_self);
        auto mat    = HBTensor<float, 2>(_mat);
        auto vec    = HBTensor<float>(_vec);
        auto result = HBTensor<float>(_result);
        auto acc    = HBTensor<float, 2>(_acc);
        float beta  = *_beta;
        float alpha = *_alpha;

        // get data pointers
        float* self_p = (float*) self.data_ptr();
        float* mat_p  = (float*) mat.data_ptr();
        float* vec_p  = (float*) vec.data_ptr();
        float* res_p  = (float*) result.data_ptr();
        float* acc_p  = (float*) acc.data_ptr();

        // Start profiling
        bsg_cuda_print_stat_kernel_start();

        // Find the size of the vector
        int vec_size = vec.dim(0);
        
        // Find the range of rows and columns this tile operates on
        int num_rows  = ((int) mat.dim(0)) / bsg_tiles_X;
        int num_cols  = ((int) mat.dim(1)) / bsg_tiles_Y;

        int row_start = (__bsg_id / bsg_tiles_Y) * num_rows;
        int row_end   = row_start + num_rows;
        row_end       = row_end > mat.dim(0) ? mat.dim(0) : row_end;
        int col_start = (__bsg_id % bsg_tiles_Y) * num_cols;
        int col_end   = col_start + num_cols;
        col_end       = col_end > mat.dim(1) ? mat.dim(1) : col_end;

        // Column in the accumulator tensor
        int col_acc   = (__bsg_id % bsg_tiles_Y);

        // Local accumulate array
        // float* acc_l  = (float*) malloc( sizeof(float) * num_rows );

        // Matrix x vector product of sub-matrix in acc matrix
        int c = col_start;
        /*
        for ( ; c+2 <= col_end; c += 2 ) {
            float vec_c0 = vec_p[c];
            float vec_c1 = vec_p[c+1];

            // float vec   = vec(c);
            int r = row_start;
            for ( ; r + 4 <= row_end; r += 4 ) {
                float mat_r0 = mat_p[r * vec_size + c];
                float mat_r1 = mat_p[(r+1) * vec_size + c];
                float mat_r2 = mat_p[(r+2) * vec_size + c];
                float mat_r3 = mat_p[(r+3) * vec_size + c];

                // acc_l[r] += mat_r0 * vec_c0;
                // acc_l[(r+1)] += mat_r1 * vec_c0;
                // acc_l[(r+2)] += mat_r2 * vec_c0;
                // acc_l[(r+3)] += mat_r3 * vec_c0;                
                
                acc_p[r * bsg_tiles_Y + col_acc] += mat_r0 * vec_c0;
                acc_p[(r+1) * bsg_tiles_Y + col_acc] += mat_r1 * vec_c0;
                acc_p[(r+2) * bsg_tiles_Y + col_acc] += mat_r2 * vec_c0;
                acc_p[(r+3) * bsg_tiles_Y + col_acc] += mat_r3 * vec_c0;
                
                // acc_p[r * bsg_tiles_Y + col_acc] += mat_p[r * vec_size + c] * vec_c;
                // acc(r, col_acc) += mat(r, c) * vec(c);
            }
            for (; r < row_end; r++ ) {
                float mat_r = mat_p[r * vec_size + c];
                // acc_l[r] += mat_r * vec_c0;
                acc_p[r * bsg_tiles_Y + col_acc] += mat_r * vec_c0;
            } 

            r = row_start;
            for ( ; r + 4 <= row_end; r += 4 ) {
                float mat_r0 = mat_p[r * vec_size + (c+1)];
                float mat_r1 = mat_p[(r+1) * vec_size + (c+1)];
                float mat_r2 = mat_p[(r+2) * vec_size + (c+1)];
                float mat_r3 = mat_p[(r+3) * vec_size + (c+1)];

                // acc_l[r] += mat_r0 * vec_c1;
                // acc_l[(r+1)] += mat_r1 * vec_c1;
                // acc_l[(r+2)] += mat_r2 * vec_c1;
                // acc_l[(r+3)] += mat_r3 * vec_c1; 
                
                acc_p[r * bsg_tiles_Y + col_acc] += mat_r0 * vec_c1;
                acc_p[(r+1) * bsg_tiles_Y + col_acc] += mat_r1 * vec_c1;
                acc_p[(r+2) * bsg_tiles_Y + col_acc] += mat_r2 * vec_c1;
                acc_p[(r+3) * bsg_tiles_Y + col_acc] += mat_r3 * vec_c1;
                
                // acc_p[r * bsg_tiles_Y + col_acc] += mat_p[r * vec_size + c] * vec_c;
                // acc(r, col_acc) += mat(r, c) * vec(c);
            }
            for (; r < row_end; r++ ) {
                float mat_r = mat_p[r * vec_size + (c+1)];
                //acc_l[r] += mat_r * vec_c1;
                acc_p[r * bsg_tiles_Y + col_acc] += mat_r * vec_c1;
            }
            
        }
        */
        for (; c < col_end; c++ ){
            float vec_c = vec_p[c];
            
            int r = row_start;
            for ( ; r + 8 <= row_end; r += 8 ) {
                float mat_r0 = mat_p[r * vec_size + c];
                float mat_r1 = mat_p[(r+1) * vec_size + c];
                float mat_r2 = mat_p[(r+2) * vec_size + c];
                float mat_r3 = mat_p[(r+3) * vec_size + c];
                float mat_r4 = mat_p[(r+4) * vec_size + c];
                float mat_r5 = mat_p[(r+5) * vec_size + c];
                float mat_r6 = mat_p[(r+6) * vec_size + c];
                float mat_r7 = mat_p[(r+7) * vec_size + c];

                // acc_l[r] += mat_r0 * vec_c;
                // acc_l[(r+1)] += mat_r1 * vec_c;
                // acc_l[(r+2)] += mat_r2 * vec_c;
                // acc_l[(r+3)] += mat_r3 * vec_c; 
                
                acc_p[r * bsg_tiles_Y + col_acc] += mat_r0 * vec_c;
                acc_p[(r+1) * bsg_tiles_Y + col_acc] += mat_r1 * vec_c;
                acc_p[(r+2) * bsg_tiles_Y + col_acc] += mat_r2 * vec_c;
                acc_p[(r+3) * bsg_tiles_Y + col_acc] += mat_r3 * vec_c;
                acc_p[(r+4) * bsg_tiles_Y + col_acc] += mat_r4 * vec_c;
                acc_p[(r+5) * bsg_tiles_Y + col_acc] += mat_r5 * vec_c;
                acc_p[(r+6) * bsg_tiles_Y + col_acc] += mat_r6 * vec_c;
                acc_p[(r+7) * bsg_tiles_Y + col_acc] += mat_r7 * vec_c;
                
                // acc_p[r * bsg_tiles_Y + col_acc] += mat_p[r * vec_size + c] * vec_c;
                // acc(r, col_acc) += mat(r, c) * vec(c);
            }
            for (; r < row_end; r++ ) {
                float mat_r = mat_p[r * vec_size + c];
                // acc_l[r] += mat_r * vec_c;
                acc_p[r * bsg_tiles_Y + col_acc] += mat_r * vec_c;
            } 

        }

        // Local accumulate to global accumulate
        //for ( int r = row_start; r < row_end; r++ ) {
        //    acc_p[r * bsg_tiles_Y + col_acc] = acc_l[r - row_start];
        //} 

        // Wait for all the tiles
        g_barrier.sync();

        // Add the acc elements to produce result (only column 0 tiles)
        if ( __bsg_id % bsg_tiles_Y == 0 ) {
            for ( int r = row_start; r < row_end; r++ ) {
                float res = 0;
                for ( int c = 0; c < bsg_tiles_Y; c++ ) {
                    res += acc_p[r * bsg_tiles_Y + c];
                    // res += acc(r, c);
                }
                res_p[r] = ( alpha * self_p[r] ) + (beta * res );
                // result(r) = ( alpha * self(r) ) + ( beta * res ); 
                // result(r) = acc(r, 0);
            }
        }

        // addmv_helper(self, mat, vec, result, alpha, beta, row_start, row_end, vec_size );
        
        // addmv_helper_p(self_p, mat_p, vec_p, res_p, alpha, beta, row_start, row_end, vec_size );

        //   End profiling
        bsg_cuda_print_stat_kernel_end();

        g_barrier.sync();
        return 0;
    }

    HB_EMUL_REG_KERNEL(tensorlib_addmv_naive, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, float*, float*)
}