//====================================================================
// addmm kernel common subroutine
// 06/20/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

// XXX: this compute subroutine handles one row of 8 outputs at a time
// to avoid dependency chains
//
// output row 1:
//   iter 0 - mat1[0][0] * ( mat2[0][0], mat2[0][1], ..., mat2[0][7] )
//   iter 1 - mat1[0][1] * ( mat2[1][0], mat2[1][1], ..., mat2[1][7] )
//   iter 2 - mat1[0][2] * ( mat2[2][0], mat2[2][1], ..., mat2[2][7] )

#ifndef _ADDMM_H_
#define _ADDMM_H_

inline void compute_simple(
          float* dest,
          float* sp_mat1,
          float* sp_mat2) {
    for (int iii = 0; iii < BLOCK_DIM; iii += 4) {
      for(int jjj = 0; jjj < BLOCK_DIM; jjj += 4) {
        int dest_base = iii * BLOCK_DIM + jjj;
        register float res00 = dest[dest_base + 0             + 0];
        register float res01 = dest[dest_base + 0             + 1];
        register float res02 = dest[dest_base + 0             + 2];
        register float res03 = dest[dest_base + 0             + 3];
        register float res10 = dest[dest_base + BLOCK_DIM     + 0];
        register float res11 = dest[dest_base + BLOCK_DIM     + 1];
        register float res12 = dest[dest_base + BLOCK_DIM     + 2];
        register float res13 = dest[dest_base + BLOCK_DIM     + 3];
        register float res20 = dest[dest_base + 2 * BLOCK_DIM + 0];
        register float res21 = dest[dest_base + 2 * BLOCK_DIM + 1];
        register float res22 = dest[dest_base + 2 * BLOCK_DIM + 2];
        register float res23 = dest[dest_base + 2 * BLOCK_DIM + 3];
        register float res30 = dest[dest_base + 3 * BLOCK_DIM + 0];
        register float res31 = dest[dest_base + 3 * BLOCK_DIM + 1];
        register float res32 = dest[dest_base + 3 * BLOCK_DIM + 2];
        register float res33 = dest[dest_base + 3 * BLOCK_DIM + 3];
        for(int kkk = 0; kkk < BLOCK_DIM; kkk++) {
          // for iiii in 0...4
          //   for jjjj in 0...4
          int mat1_base = kkk + iii * BLOCK_DIM;
          register float mat1_0 = sp_mat1[mat1_base + 0];
          register float mat1_1 = sp_mat1[mat1_base + BLOCK_DIM];
          register float mat1_2 = sp_mat1[mat1_base + 2 * BLOCK_DIM];
          register float mat1_3 = sp_mat1[mat1_base + 3 * BLOCK_DIM];
          int mat2_base = kkk * BLOCK_DIM + jjj;
          register float mat2_0 = sp_mat2[mat2_base + 0];
          register float mat2_1 = sp_mat2[mat2_base + 1];
          register float mat2_2 = sp_mat2[mat2_base + 2];
          register float mat2_3 = sp_mat2[mat2_base + 3];
          // compute
          res00 += mat1_0 * mat2_0;
          res01 += mat1_0 * mat2_1;
          res02 += mat1_0 * mat2_2;
          res03 += mat1_0 * mat2_3;
          res10 += mat1_1 * mat2_0;
          res11 += mat1_1 * mat2_1;
          res12 += mat1_1 * mat2_2;
          res13 += mat1_1 * mat2_3;
          res20 += mat1_2 * mat2_0;
          res21 += mat1_2 * mat2_1;
          res22 += mat1_2 * mat2_2;
          res23 += mat1_2 * mat2_3;
          res30 += mat1_3 * mat2_0;
          res31 += mat1_3 * mat2_1;
          res32 += mat1_3 * mat2_2;
          res33 += mat1_3 * mat2_3;
        }
        dest[dest_base + 0             + 0] = res00;
        dest[dest_base + 0             + 1] = res01;
        dest[dest_base + 0             + 2] = res02;
        dest[dest_base + 0             + 3] = res03;
        dest[dest_base + BLOCK_DIM     + 0] = res10;
        dest[dest_base + BLOCK_DIM     + 1] = res11;
        dest[dest_base + BLOCK_DIM     + 2] = res12;
        dest[dest_base + BLOCK_DIM     + 3] = res13;
        dest[dest_base + 2 * BLOCK_DIM + 0] = res20;
        dest[dest_base + 2 * BLOCK_DIM + 1] = res21;
        dest[dest_base + 2 * BLOCK_DIM + 2] = res22;
        dest[dest_base + 2 * BLOCK_DIM + 3] = res23;
        dest[dest_base + 3 * BLOCK_DIM + 0] = res30;
        dest[dest_base + 3 * BLOCK_DIM + 1] = res31;
        dest[dest_base + 3 * BLOCK_DIM + 2] = res32;
        dest[dest_base + 3 * BLOCK_DIM + 3] = res33;
      }
    }
}


// XXX: in the test case, 1024x32 @ 32x1024 case, there is no partial blocks

inline void compute(
          float* dest,
          float* sp_mat1,
          float* sp_mat2,
          int dim_y,
          int dim_x,
          int mid_dim) {
    for (int i = 0; i < dim_y; i++) {
        int dest_row_offset = i * dim_x;
        int mat1_row_offset = i * mid_dim;
        for (int j = 0; j < dim_x; j++) {
            int k = 0;
            register float tmp_fix = 0.0f;
            for (;k < mid_dim; k++) {
                int mat1_idx = mat1_row_offset + k;
                int mat2_idx = k * dim_x + j;
                tmp_fix += sp_mat1[mat1_idx] * sp_mat2[mat2_idx];
            }
            dest[dest_row_offset + j] += tmp_fix;
        }
    }
}

// XXX: to get good performance, we assume matrix is contiguous in
// memory

inline void dram_to_sp_simple(
          float* dest,
          HBTensor<float, 2> src,
          int r_idx,
          int c_idx) {
    float* src_ptr = (float*)src.data_ptr();
    uint32_t* src_strides = src.get_strides();
    float* src_base = src_ptr + (r_idx * BLOCK_DIM * src_strides[0])
                      + (c_idx * BLOCK_DIM * src_strides[1]);
    int row_offset = 0;
    for (int i = 0; i < BLOCK_DIM; i++) {
        float* src_offset = src_base;
        if (BLOCK_DIM == 12) {
            register float tmp0 = *(src_offset + 0);
            register float tmp1 = *(src_offset + 1);
            register float tmp2 = *(src_offset + 2);
            register float tmp3 = *(src_offset + 3);
            register float tmp4 = *(src_offset + 4);
            register float tmp5 = *(src_offset + 5);
            register float tmp6 = *(src_offset + 6);
            register float tmp7 = *(src_offset + 7);
            register float tmp8 = *(src_offset + 8);
            register float tmp9 = *(src_offset + 9);
            register float tmp10 = *(src_offset + 10);
            register float tmp11 = *(src_offset + 11);
            asm volatile("": : :"memory");
            dest[row_offset + 0] = tmp0;
            dest[row_offset + 1] = tmp1;
            dest[row_offset + 2] = tmp2;
            dest[row_offset + 3] = tmp3;
            dest[row_offset + 4] = tmp4;
            dest[row_offset + 5] = tmp5;
            dest[row_offset + 6] = tmp6;
            dest[row_offset + 7] = tmp7;
            dest[row_offset + 8] = tmp8;
            dest[row_offset + 9] = tmp9;
            dest[row_offset + 10] = tmp10;
            dest[row_offset + 11] = tmp11;
            row_offset += 12;
        }
        else {
            for (int j = 0; j < BLOCK_DIM; j += 8) {
                register float tmp0 = *(src_offset + 0);
                register float tmp1 = *(src_offset + 1);
                register float tmp2 = *(src_offset + 2);
                register float tmp3 = *(src_offset + 3);
                register float tmp4 = *(src_offset + 4);
                register float tmp5 = *(src_offset + 5);
                register float tmp6 = *(src_offset + 6);
                register float tmp7 = *(src_offset + 7);
                asm volatile("": : :"memory");
                dest[row_offset + 0] = tmp0;
                dest[row_offset + 1] = tmp1;
                dest[row_offset + 2] = tmp2;
                dest[row_offset + 3] = tmp3;
                dest[row_offset + 4] = tmp4;
                dest[row_offset + 5] = tmp5;
                dest[row_offset + 6] = tmp6;
                dest[row_offset + 7] = tmp7;
                src_offset += 8;
                row_offset += 8;
            }
        }
        src_base += src_strides[0];
    }
}

inline void sp_to_dram(
          HBTensor<float, 2> dest,
          float* src,
          int r_idx,
          int c_idx) {
    float* dest_ptr = (float*)dest.data_ptr();
    uint32_t* dest_strides = dest.get_strides();
    float* dest_base = dest_ptr + (r_idx * BLOCK_DIM * dest_strides[0])
                       + (c_idx * BLOCK_DIM * dest_strides[1]);
    int row_offset = 0;
    for (int i = 0; i < BLOCK_DIM; i++) {
        float* dest_offset = dest_base;
        if (BLOCK_DIM == 12) {
            register float tmp0 = src[row_offset + 0];
            register float tmp1 = src[row_offset + 1];
            register float tmp2 = src[row_offset + 2];
            register float tmp3 = src[row_offset + 3];
            register float tmp4 = src[row_offset + 4];
            register float tmp5 = src[row_offset + 5];
            register float tmp6 = src[row_offset + 6];
            register float tmp7 = src[row_offset + 7];
            register float tmp8 = src[row_offset + 8];
            register float tmp9 = src[row_offset + 9];
            register float tmp10 = src[row_offset + 10];
            register float tmp11 = src[row_offset + 11];
            asm volatile("": : :"memory");
            *(dest_offset + 0) = tmp0;
            *(dest_offset + 1) = tmp1;
            *(dest_offset + 2) = tmp2;
            *(dest_offset + 3) = tmp3;
            *(dest_offset + 4) = tmp4;
            *(dest_offset + 5) = tmp5;
            *(dest_offset + 6) = tmp6;
            *(dest_offset + 7) = tmp7;
            *(dest_offset + 8) = tmp8;
            *(dest_offset + 9) = tmp9;
            *(dest_offset + 10) = tmp10;
            *(dest_offset + 11) = tmp11;
            row_offset += 12;
        }
        else {
            for (int j = 0; j < BLOCK_DIM; j += 8) {
                register float tmp0 = src[row_offset + 0];
                register float tmp1 = src[row_offset + 1];
                register float tmp2 = src[row_offset + 2];
                register float tmp3 = src[row_offset + 3];
                register float tmp4 = src[row_offset + 4];
                register float tmp5 = src[row_offset + 5];
                register float tmp6 = src[row_offset + 6];
                register float tmp7 = src[row_offset + 7];
                asm volatile("": : :"memory");
                *(dest_offset + 0) = tmp0;
                *(dest_offset + 1) = tmp1;
                *(dest_offset + 2) = tmp2;
                *(dest_offset + 3) = tmp3;
                *(dest_offset + 4) = tmp4;
                *(dest_offset + 5) = tmp5;
                *(dest_offset + 6) = tmp6;
                *(dest_offset + 7) = tmp7;
                dest_offset += 8;
                row_offset += 8;
            }
        }
        dest_base += dest_strides[0];
    }
}

inline void addmm_and_sp_to_dram(
          HBTensor<float, 2> dest,
          float* src,
          float* input,
          float alpha,
          float beta,
          int r_idx,
          int c_idx) {
    float* dest_ptr = (float*)dest.data_ptr();
    uint32_t* dest_strides = dest.get_strides();
    float* dest_base = dest_ptr + (r_idx * BLOCK_DIM * dest_strides[0])
                       + (c_idx * BLOCK_DIM * dest_strides[1]);
    int row_offset = 0;
    for (int i = 0; i < BLOCK_DIM; i++) {
        float* dest_offset = dest_base;
        if (BLOCK_DIM == 12) {
            register float tmp0 = src[row_offset + 0];
            register float tmp1 = src[row_offset + 1];
            register float tmp2 = src[row_offset + 2];
            register float tmp3 = src[row_offset + 3];
            register float tmp4 = src[row_offset + 4];
            register float tmp5 = src[row_offset + 5];
            register float tmp6 = src[row_offset + 6];
            register float tmp7 = src[row_offset + 7];
            register float tmp8 = src[row_offset + 8];
            register float tmp9 = src[row_offset + 9];
            register float tmp10 = src[row_offset + 10];
            register float tmp11 = src[row_offset + 11];
            register float input0 = beta * input[row_offset + 0];
            register float input1 = beta * input[row_offset + 1];
            register float input2 = beta * input[row_offset + 2];
            register float input3 = beta * input[row_offset + 3];
            register float input4 = beta * input[row_offset + 4];
            register float input5 = beta * input[row_offset + 5];
            register float input6 = beta * input[row_offset + 6];
            register float input7 = beta * input[row_offset + 7];
            register float input8 = beta * input[row_offset + 8];
            register float input9 = beta * input[row_offset + 9];
            register float input10 = beta * input[row_offset + 10];
            register float input11 = beta * input[row_offset + 11];
            asm volatile("": : :"memory");
            *(dest_offset + 0) = alpha * tmp0 + input0;
            *(dest_offset + 1) = alpha * tmp1 + input1;
            *(dest_offset + 2) = alpha * tmp2 + input2;
            *(dest_offset + 3) = alpha * tmp3 + input3;
            *(dest_offset + 4) = alpha * tmp4 + input4;
            *(dest_offset + 5) = alpha * tmp5 + input5;
            *(dest_offset + 6) = alpha * tmp6 + input6;
            *(dest_offset + 7) = alpha * tmp7 + input7;
            *(dest_offset + 8) = alpha * tmp8 + input8;
            *(dest_offset + 9) = alpha * tmp9 + input9;
            *(dest_offset + 10) = alpha * tmp10 + input10;
            *(dest_offset + 11) = alpha * tmp11 + input11;
            row_offset += 12;
        }
        else {
            for (int j = 0; j < BLOCK_DIM; j += 8) {
                register float tmp0 = src[row_offset + 0];
                register float tmp1 = src[row_offset + 1];
                register float tmp2 = src[row_offset + 2];
                register float tmp3 = src[row_offset + 3];
                register float tmp4 = src[row_offset + 4];
                register float tmp5 = src[row_offset + 5];
                register float tmp6 = src[row_offset + 6];
                register float tmp7 = src[row_offset + 7];
                register float input0 = beta * input[row_offset + 0];
                register float input1 = beta * input[row_offset + 1];
                register float input2 = beta * input[row_offset + 2];
                register float input3 = beta * input[row_offset + 3];
                register float input4 = beta * input[row_offset + 4];
                register float input5 = beta * input[row_offset + 5];
                register float input6 = beta * input[row_offset + 6];
                register float input7 = beta * input[row_offset + 7];
                asm volatile("": : :"memory");
                *(dest_offset + 0) = alpha * tmp0 + input0;
                *(dest_offset + 1) = alpha * tmp1 + input1;
                *(dest_offset + 2) = alpha * tmp2 + input2;
                *(dest_offset + 3) = alpha * tmp3 + input3;
                *(dest_offset + 4) = alpha * tmp4 + input4;
                *(dest_offset + 5) = alpha * tmp5 + input5;
                *(dest_offset + 6) = alpha * tmp6 + input6;
                *(dest_offset + 7) = alpha * tmp7 + input7;
                dest_offset += 8;
                row_offset += 8;
            }
        }
        dest_base += dest_strides[0];
    }
}

inline void addmm_and_sp_to_dram_naive(
          HBTensor<float, 2> dest,
          float* src,
          float* input,
          int r_idx,
          int c_idx) {
    float* dest_ptr = (float*)dest.data_ptr();
    uint32_t* dest_strides = dest.get_strides();
    float* dest_base = dest_ptr + (r_idx * BLOCK_DIM * dest_strides[0])
                       + (c_idx * BLOCK_DIM * dest_strides[1]);
    int row_offset = 0;
    for (int i = 0; i < BLOCK_DIM; i++) {
        float* dest_offset = dest_base;
        if (BLOCK_DIM == 12) {
            register float tmp0 = src[row_offset + 0];
            register float tmp1 = src[row_offset + 1];
            register float tmp2 = src[row_offset + 2];
            register float tmp3 = src[row_offset + 3];
            register float tmp4 = src[row_offset + 4];
            register float tmp5 = src[row_offset + 5];
            register float tmp6 = src[row_offset + 6];
            register float tmp7 = src[row_offset + 7];
            register float tmp8 = src[row_offset + 8];
            register float tmp9 = src[row_offset + 9];
            register float tmp10 = src[row_offset + 10];
            register float tmp11 = src[row_offset + 11];
            register float input0 = input[row_offset + 0];
            register float input1 = input[row_offset + 1];
            register float input2 = input[row_offset + 2];
            register float input3 = input[row_offset + 3];
            register float input4 = input[row_offset + 4];
            register float input5 = input[row_offset + 5];
            register float input6 = input[row_offset + 6];
            register float input7 = input[row_offset + 7];
            register float input8 = input[row_offset + 8];
            register float input9 = input[row_offset + 9];
            register float input10 = input[row_offset + 10];
            register float input11 = input[row_offset + 11];
            asm volatile("": : :"memory");
            *(dest_offset + 0) = tmp0 + input0;
            *(dest_offset + 1) = tmp1 + input1;
            *(dest_offset + 2) = tmp2 + input2;
            *(dest_offset + 3) = tmp3 + input3;
            *(dest_offset + 4) = tmp4 + input4;
            *(dest_offset + 5) = tmp5 + input5;
            *(dest_offset + 6) = tmp6 + input6;
            *(dest_offset + 7) = tmp7 + input7;
            *(dest_offset + 8) = tmp8 + input8;
            *(dest_offset + 9) = tmp9 + input9;
            *(dest_offset + 10) = tmp10 + input10;
            *(dest_offset + 11) = tmp11 + input11;
            row_offset += 12;
        }
        else {
            for (int j = 0; j < BLOCK_DIM; j += 8) {
                register float tmp0 = src[row_offset + 0];
                register float tmp1 = src[row_offset + 1];
                register float tmp2 = src[row_offset + 2];
                register float tmp3 = src[row_offset + 3];
                register float tmp4 = src[row_offset + 4];
                register float tmp5 = src[row_offset + 5];
                register float tmp6 = src[row_offset + 6];
                register float tmp7 = src[row_offset + 7];
                register float input0 = input[row_offset + 0];
                register float input1 = input[row_offset + 1];
                register float input2 = input[row_offset + 2];
                register float input3 = input[row_offset + 3];
                register float input4 = input[row_offset + 4];
                register float input5 = input[row_offset + 5];
                register float input6 = input[row_offset + 6];
                register float input7 = input[row_offset + 7];
                asm volatile("": : :"memory");
                *(dest_offset + 0) = tmp0 + input0;
                *(dest_offset + 1) = tmp1 + input1;
                *(dest_offset + 2) = tmp2 + input2;
                *(dest_offset + 3) = tmp3 + input3;
                *(dest_offset + 4) = tmp4 + input4;
                *(dest_offset + 5) = tmp5 + input5;
                *(dest_offset + 6) = tmp6 + input6;
                *(dest_offset + 7) = tmp7 + input7;
                dest_offset += 8;
                row_offset += 8;
            }
        }
        dest_base += dest_strides[0];
    }
}

inline void reset_sp(float* dest) {
  // initialize scratchpad (init to 0's)
  for (int sp = 0; sp < BLOCK_DIM * BLOCK_DIM; sp += 16) {
      dest[sp +  0] = 0;
      dest[sp +  1] = 0;
      dest[sp +  2] = 0;
      dest[sp +  3] = 0;
      dest[sp +  4] = 0;
      dest[sp +  5] = 0;
      dest[sp +  6] = 0;
      dest[sp +  7] = 0;
      dest[sp +  8] = 0;
      dest[sp +  9] = 0;
      dest[sp + 10] = 0;
      dest[sp + 11] = 0;
      dest[sp + 12] = 0;
      dest[sp + 13] = 0;
      dest[sp + 14] = 0;
      dest[sp + 15] = 0;
  }
}

#endif
