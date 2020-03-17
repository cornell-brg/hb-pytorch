#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/hammerblade/OffloadDef.h>

namespace at {
namespace native {

//============================================
// TensorIterator operations with zero scalar
//============================================

void offload_op_nullary(TensorIterator& iter, const char* kernel);
void offload_op_unary(TensorIterator& iter, const char* kernel);
void offload_op_binary(TensorIterator& iter, const char* kernel);

//============================================
// TensorIterator operations with one scalar
//============================================

void offload_op_nullary(TensorIterator& iter, Scalar alpha,
                        const char* kernel);
void offload_op_unary(TensorIterator& iter, Scalar alpha,
                        const char* kernel);
void offload_op_binary(TensorIterator& iter, Scalar alpha,
                        const char* kernel);

//============================================
// TensorIterator operations with two scalars
//============================================

void offload_op_nullary(TensorIterator& iter, Scalar beta,
                        Scalar alpha, const char* kernel);
void offload_op_unary(TensorIterator& iter, Scalar beta,
                        Scalar alpha, const char* kernel);
void offload_op_binary(TensorIterator& iter, Scalar beta,
                        Scalar alpha, const char* kernel);

} // namespace native
} // namespace at
