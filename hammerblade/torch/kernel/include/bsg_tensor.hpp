#ifndef _BSG_TENSOR_HPP
#define _BSG_TENSOR_HPP

typedef struct {
  uint32_t N;
  uint32_t dims;
  uint32_t* strides;
  float* data;
} bsg_tensor_t;

#endif // _BSG_TENSOR_HPP
