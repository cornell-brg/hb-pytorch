#pragma once

#include <ATen/ATen.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at {
namespace native {

Tensor& hb_convolution_nogroup(
    const Tensor& input, const Tensor& weight, 
    const Tensor& bias, IntArrayRef stride, 
    IntArrayRef padding, IntArrayRef dilation, 
    bool transposed, IntArrayRef output_padding);

}} // namespace at::native
