#ifndef _BSG_TENSOR_HPP
#define _BSG_TENSOR_HPP

typedef struct {
  uint32_t N;
  uint32_t dims;
#ifdef HB_EMUL
  uint64_t strides;
  uint64_t data;
#else
  uint32_t strides;
  uint32_t data;
#endif
} bsg_tensor_t;

typedef struct {
  uint32_t N;
  uint32_t data;
} bsg_vector_t;

#endif // _BSG_TENSOR_HPP
