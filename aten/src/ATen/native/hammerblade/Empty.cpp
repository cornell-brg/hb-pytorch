// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/CPUGenerator.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Deprecated.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/TensorOptions.h>
#include <TH/THAllocator.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>
#include <ATen/hammerblade/HammerBladeContext.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor empty_hb(IntArrayRef size, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ASSERT(options.device().type() == DeviceType::HAMMERBLADE);
  TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  check_size_nonnegative(size);

  c10::Allocator* allocator = at::hammerblade::getHammerBladeDeviceAllocator();

  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
    dtype,
    nelements,
    allocator->allocate(nelements * dtype.itemsize()),
    allocator,
    /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(std::move(storage_impl), at::TensorTypeId::HammerBladeTensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = optional_memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

}} // at::native
