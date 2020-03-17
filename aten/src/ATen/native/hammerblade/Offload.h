#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>
#include <ATen/native/hammerblade/OffloadIter.h>

namespace at {
namespace native {

void offload_memcpy(eva_t dest, eva_t src, uint32_t n);

void offload_tensor_scalar_impl(std::vector<Tensor> tensors, std::vector<Scalar> scalars,
                                const char* kernel);
#define HB_OFFLOAD_TENSOR_KERNEL_GET_MACRO(_1,_2,_3,_4,_5,NAME,...) NAME
#define HB_OFFLOAD_TENSOR_KERNEL(...) HB_OFFLOAD_TENSOR_KERNEL_GET_MACRO(__VA_ARGS__,                   \
                                                                    HB_OFFLOAD_TENSOR_KERNEL_5ARGS,     \
                                                                    HB_OFFLOAD_TENSOR_KERNEL_4ARGS,     \
                                                                    HB_OFFLOAD_TENSOR_KERNEL_3ARGS,     \
                                                                    HB_OFFLOAD_TENSOR_KERNEL_2ARGS,     \
                                                                    HB_OFFLOAD_TENSOR_KERNEL_1ARGS)     \
                                                                    (__VA_ARGS__)                       \

#define HB_OFFLOAD_TENSOR_KERNEL_1ARGS(kernel)                                                          \
do {                                                                                                    \
  std::vector<Tensor> args;                                                                             \
  std::vector<Scalar> scalars;                                                                          \
  offload_tensor_scalar_impl(args, scalars, kernel);                                                    \
} while (0);

#define HB_OFFLOAD_TENSOR_KERNEL_2ARGS(arg0, kernel)                                                    \
do {                                                                                                    \
  std::vector<Tensor> args;                                                                             \
  args.push_back(arg0);                                                                                 \
  std::vector<Scalar> scalars;                                                                          \
  offload_tensor_scalar_impl(args, scalars, kernel);                                                    \
} while (0);

#define HB_OFFLOAD_TENSOR_KERNEL_3ARGS(arg0, arg1, kernel)                                              \
do {                                                                                                    \
  std::vector<Tensor> args;                                                                             \
  args.push_back(arg0);                                                                                 \
  args.push_back(arg1);                                                                                 \
  std::vector<Scalar> scalars;                                                                          \
  offload_tensor_scalar_impl(args, scalars, kernel);                                                    \
} while (0);

#define HB_OFFLOAD_TENSOR_KERNEL_4ARGS(arg0, arg1, arg2, kernel)                                        \
do {                                                                                                    \
  std::vector<Tensor> args;                                                                             \
  args.push_back(arg0);                                                                                 \
  args.push_back(arg1);                                                                                 \
  args.push_back(arg2);                                                                                 \
  std::vector<Scalar> scalars;                                                                          \
  offload_tensor_scalar_impl(args, scalars, kernel);                                                    \
} while (0);

#define HB_OFFLOAD_TENSOR_KERNEL_5ARGS(arg0, arg1, arg2, arg3, kernel)                                  \
do {                                                                                                    \
  std::vector<Tensor> args;                                                                             \
  args.push_back(arg0);                                                                                 \
  args.push_back(arg1);                                                                                 \
  args.push_back(arg2);                                                                                 \
  args.push_back(arg3);                                                                                 \
  std::vector<Scalar> scalars;                                                                          \
  offload_tensor_scalar_impl(args, scalars, kernel);                                                    \
} while (0);

void offload_convolution_forward(Tensor& output, const Tensor& input,
    const Tensor& weight, IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups);

} // namespace native
} // namespace at
