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

inline void compute_simple(
          float* dest,
          float* sp_mat1,
          float* sp_mat2,
          int dim_y,
          int dim_x,
          int mid_dim) {
    for (int i = 0; i < BLOCK_DIM; i++) {
        int dest_row_offset = i * BLOCK_DIM;
        int mat1_row_offset = i * BLOCK_DIM;
        register float res0 = dest[dest_row_offset + 0];
        register float res1 = dest[dest_row_offset + 1];
        register float res2 = dest[dest_row_offset + 2];
        register float res3 = dest[dest_row_offset + 3];
        register float res4 = dest[dest_row_offset + 4];
        register float res5 = dest[dest_row_offset + 5];
        register float res6 = dest[dest_row_offset + 6];
        register float res7 = dest[dest_row_offset + 7];
        int mat2_row_offset = 0;
        for (int k = 0; k < BLOCK_DIM; k++) {
            register float mat1 = sp_mat1[mat1_row_offset + k];
            res0 += (mat1 * sp_mat2[mat2_row_offset + 0]);
            res1 += (mat1 * sp_mat2[mat2_row_offset + 1]);
            res2 += (mat1 * sp_mat2[mat2_row_offset + 2]);
            res3 += (mat1 * sp_mat2[mat2_row_offset + 3]);
            res4 += (mat1 * sp_mat2[mat2_row_offset + 4]);
            res5 += (mat1 * sp_mat2[mat2_row_offset + 5]);
            res6 += (mat1 * sp_mat2[mat2_row_offset + 6]);
            res7 += (mat1 * sp_mat2[mat2_row_offset + 7]);
            mat2_row_offset += BLOCK_DIM;
        }
        dest[dest_row_offset + 0] = res0;
        dest[dest_row_offset + 1] = res1;
        dest[dest_row_offset + 2] = res2;
        dest[dest_row_offset + 3] = res3;
        dest[dest_row_offset + 4] = res4;
        dest[dest_row_offset + 5] = res5;
        dest[dest_row_offset + 6] = res6;
        dest[dest_row_offset + 7] = res7;
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

// XXX: to get good performance, we assume BLOCK_DIM === 8, and the matrix is contiguous in
// memory

inline void dram_to_sp_simple(
          float* dest,
          HBTensor<float, 2> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    float* src_ptr = (float*)src.data_ptr();
    uint32_t* src_strides = src.get_strides();
    float* src_base = src_ptr + (r_idx * BLOCK_DIM * src_strides[0])
                      + (c_idx * BLOCK_DIM * src_strides[1]);
    int row_offset = 0;
    for (int i = 0; i < dim_y; i++) {
        register float tmp0 = *(src_base + 0);
        register float tmp1 = *(src_base + 1);
        register float tmp2 = *(src_base + 2);
        register float tmp3 = *(src_base + 3);
        register float tmp4 = *(src_base + 4);
        register float tmp5 = *(src_base + 5);
        register float tmp6 = *(src_base + 6);
        register float tmp7 = *(src_base + 7);
        asm volatile("": : :"memory");
        dest[row_offset + 0] = tmp0;
        dest[row_offset + 1] = tmp1;
        dest[row_offset + 2] = tmp2;
        dest[row_offset + 3] = tmp3;
        dest[row_offset + 4] = tmp4;
        dest[row_offset + 5] = tmp5;
        dest[row_offset + 6] = tmp6;
        dest[row_offset + 7] = tmp7;
        src_base += src_strides[0];
        row_offset += dim_x;
    }
}

// XXX: in the test case, 1024x32 @ 32x1024 case, there is no partial blocks

inline void dram_to_sp(
          float* dest,
          HBTensor<float, 2> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    for (int i = 0; i < dim_y; i++) {
        int row_offset = i * dim_x;
        int j = 0;
        for (;j < dim_x - 8; j += 8) {
            register float tmp0 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
            register float tmp1 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 1);
            register float tmp2 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 2);
            register float tmp3 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 3);
            register float tmp4 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 4);
            register float tmp5 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 5);
            register float tmp6 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 6);
            register float tmp7 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 7);
            asm volatile("": : :"memory");
            dest[row_offset + j]     = tmp0;
            dest[row_offset + j + 1] = tmp1;
            dest[row_offset + j + 2] = tmp2;
            dest[row_offset + j + 3] = tmp3;
            dest[row_offset + j + 4] = tmp4;
            dest[row_offset + j + 5] = tmp5;
            dest[row_offset + j + 6] = tmp6;
            dest[row_offset + j + 7] = tmp7;
        }
        // fixup
        for (;j < dim_x; j++) {
            dest[row_offset + j] = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
        }
    }
}

// same as the dram_to_sp above but with coeff

static void dram_to_sp_simple(
          float* dest,
          float coeff,
          HBTensor<float, 2> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    for (int i = 0; i < dim_y; i++) {
        int row_offset = i * dim_x;
        int j = 0;
        for (;j < dim_x; j += 8) {
            register float tmp0 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
            register float tmp1 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 1);
            register float tmp2 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 2);
            register float tmp3 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 3);
            register float tmp4 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 4);
            register float tmp5 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 5);
            register float tmp6 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 6);
            register float tmp7 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 7);
            asm volatile("": : :"memory");
            tmp0 = tmp0 * coeff;
            tmp1 = tmp1 * coeff;
            tmp2 = tmp2 * coeff;
            tmp3 = tmp3 * coeff;
            tmp4 = tmp4 * coeff;
            tmp5 = tmp5 * coeff;
            tmp6 = tmp6 * coeff;
            tmp7 = tmp7 * coeff;
            dest[row_offset + j]     = tmp0;
            dest[row_offset + j + 1] = tmp1;
            dest[row_offset + j + 2] = tmp2;
            dest[row_offset + j + 3] = tmp3;
            dest[row_offset + j + 4] = tmp4;
            dest[row_offset + j + 5] = tmp5;
            dest[row_offset + j + 6] = tmp6;
            dest[row_offset + j + 7] = tmp7;
        }
    }
}

static void dram_to_sp(
          float* dest,
          float coeff,
          HBTensor<float, 2> src,
          int dim_y,
          int dim_x,
          int r_idx,
          int c_idx) {
    for (int i = 0; i < dim_y; i++) {
        int row_offset = i * dim_x;
        int j = 0;
        for (;j < dim_x - 8; j += 8) {
            register float tmp0 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
            register float tmp1 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 1);
            register float tmp2 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 2);
            register float tmp3 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 3);
            register float tmp4 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 4);
            register float tmp5 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 5);
            register float tmp6 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 6);
            register float tmp7 = src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j + 7);
            asm volatile("": : :"memory");
            tmp0 = tmp0 * coeff;
            tmp1 = tmp1 * coeff;
            tmp2 = tmp2 * coeff;
            tmp3 = tmp3 * coeff;
            tmp4 = tmp4 * coeff;
            tmp5 = tmp5 * coeff;
            tmp6 = tmp6 * coeff;
            tmp7 = tmp7 * coeff;
            dest[row_offset + j]     = tmp0;
            dest[row_offset + j + 1] = tmp1;
            dest[row_offset + j + 2] = tmp2;
            dest[row_offset + j + 3] = tmp3;
            dest[row_offset + j + 4] = tmp4;
            dest[row_offset + j + 5] = tmp5;
            dest[row_offset + j + 6] = tmp6;
            dest[row_offset + j + 7] = tmp7;
        }
        // fixup
        for (;j < dim_x; j++) {
            dest[row_offset + j] = coeff * src(r_idx * BLOCK_DIM + i, c_idx * BLOCK_DIM + j);
        }
    }
}

