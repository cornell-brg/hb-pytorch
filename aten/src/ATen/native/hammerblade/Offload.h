#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pool.h>

#include <ATen/native/hammerblade/OffloadDef.h>
#include <ATen/native/hammerblade/OffloadUtils.h>
#include <ATen/native/hammerblade/OffloadIter.h>

namespace at {
namespace native {

//==============================================
// memcpy offloader
//==============================================

void offload_memcpy(eva_t dest, eva_t src, uint32_t n);
void offload_memcpy(TensorIterator& iter);

//==============================================
// Offloading helper which takes a list of tensors
// and a list of scalars
//==============================================

void offload_tensor_scalar_impl(std::vector<Tensor> tensors, std::vector<eva_t> device_scalars,
                                const char* kernel);

//==============================================
// hb_offload_kernel
//
// Tensors args, Scalar args, kernel name
//==============================================

//----------------------------------------------
// 0-5 tensors + 0 scalar
//----------------------------------------------

inline void hb_offload_kernel(const char* kernel) {
  std::vector<Tensor> args;
  std::vector<eva_t> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  std::vector<eva_t> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  std::vector<eva_t> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  std::vector<eva_t> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  std::vector<eva_t> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              Tensor t4, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  args.push_back(t4);
  std::vector<eva_t> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              Tensor t4, Tensor t5, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  args.push_back(t4);
  args.push_back(t5);
  std::vector<eva_t> scalars;
  offload_tensor_scalar_impl(args, scalars, kernel);
}

//----------------------------------------------
// 0-5 tensors + 1 scalar
//----------------------------------------------

template <typename ST0>
inline void hb_offload_kernel(ST0 s0, const char* kernel) {
  std::vector<Tensor> args;
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0>
inline void hb_offload_kernel(Tensor t0, ST0 s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0>
inline void hb_offload_kernel(Tensor t0, Tensor t1, ST0 s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, ST0 s0,
                              const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              ST0 s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              Tensor t4, ST0 s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  args.push_back(t4);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              Tensor t4, Tensor t5, ST0 s0, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  args.push_back(t4);
  args.push_back(t5);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

//----------------------------------------------
// 0-5 tensors + 2 scalar
//----------------------------------------------

template <typename ST0, typename ST1>
inline void hb_offload_kernel(ST0 s0, ST1 s1, const char* kernel) {
  std::vector<Tensor> args;
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  scalars.push_back(create_device_scalar(s1));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0, typename ST1>
inline void hb_offload_kernel(Tensor t0, ST0 s0, ST1 s1, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  scalars.push_back(create_device_scalar(s1));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0, typename ST1>
inline void hb_offload_kernel(Tensor t0, Tensor t1, ST0 s0,
                              ST1 s1, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  scalars.push_back(create_device_scalar(s1));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0, typename ST1>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, ST0 s0,
                              ST1 s1, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  scalars.push_back(create_device_scalar(s1));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0, typename ST1>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              ST0 s0, ST1 s1, const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  scalars.push_back(create_device_scalar(s1));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0, typename ST1>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              Tensor t4, ST0 s0, ST1 s1,
                              const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  args.push_back(t4);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  scalars.push_back(create_device_scalar(s1));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

template <typename ST0, typename ST1>
inline void hb_offload_kernel(Tensor t0, Tensor t1, Tensor t2, Tensor t3,
                              Tensor t4, Tensor t5, ST0 s0, ST1 s1,
                              const char* kernel) {
  std::vector<Tensor> args;
  args.push_back(t0);
  args.push_back(t1);
  args.push_back(t2);
  args.push_back(t3);
  args.push_back(t4);
  args.push_back(t5);
  std::vector<eva_t> scalars;
  scalars.push_back(create_device_scalar(s0));
  scalars.push_back(create_device_scalar(s1));
  offload_tensor_scalar_impl(args, scalars, kernel);
}

} // namespace native
} // namespace at
