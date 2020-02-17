#ifndef __MATRIX_MULTIPLY_HPP
#define __MATRIX_MULTIPLY_HPP
#include <cstdint>

typedef struct {
  uint32_t N;
  uint32_t dims;
  uint32_t* strides;
  float* data;
} bsg_tensor_t;

/*
 * This is the most basic single tile version of vector addition
 * that adds two vectors A and B and stores the result in C.
 *
 * The code assumes a sinlge 1x1 grid of a single 1x1 tile group
 * Do NOT use this version with larger tile groups
 */
template <typename TA, typename TB, typename TC, typename TD>
int __attribute__ ((noinline)) kernel_tile_vector_add(TA *A, TB *B, TC *C, TD alpha,
                      uint32_t WIDTH) {
        // A single tile performs the entire vector addition
	for (int iter_x = 0; iter_x < WIDTH; iter_x += 1) {
                C[iter_x] = A[iter_x] + alpha * B[iter_x];
	}

        return 0;
}

#endif
