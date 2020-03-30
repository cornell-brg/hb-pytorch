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

//==============================================
// memcpy offloader
//==============================================

void offload_memcpy(eva_t dest, eva_t src, uint32_t n);

//==============================================
// Offloading helper which takes a list of tensors
// and a list of scalars
//==============================================

void offload_tensor_scalar_impl(std::vector<Tensor> tensors, std::vector<Scalar> scalars,
                                const char* kernel);

//==============================================
// hb_offload_kernel
//
// Tensors args, Scalar args, kernel name
//==============================================

//----------------------------------------------
// 0-3 tensors + 0 scalar
//----------------------------------------------

inline void hb_offload_kernel(const char* kernel) {
  std::vector<Tensor> args;
  std::vector<Scalar> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  std::vector<Scalar> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  std::vector<Scalar> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  std::vector<Scalar> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  std::vector<Scalar> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

//----------------------------------------------
// 0-3 tensors + 1 scalar
//----------------------------------------------

inline void hb_offload_kernel(Scalar s0, const char* kernel) {
  std::vector<Tensor> args;
  std::vector<Scalar> scalars;
  scalars.push_back(s0);
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Scalar s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  std::vector<Scalar> scalars;
  scalars.push_back(s0);
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Scalar s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  std::vector<Scalar> scalars;
  scalars.push_back(s0);
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Scalar s0,
                              const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  std::vector<Scalar> scalars;
  scalars.push_back(s0);
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              Scalar s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  std::vector<Scalar> scalars;
  scalars.push_back(s0);
  offload_tensor_scalar_impl(args, scalars, kernel);
}

//==============================================
// conv offloader
//==============================================

void offload_convolution_forward(Tensor& output, const Tensor& input,
    const Tensor& weight, IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups);

void offload_convolution_add_bias(const Tensor& output, const Tensor& ibias);

} // namespace native
} // namespace at
