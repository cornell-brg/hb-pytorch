#include <ATen/native/Indexing.h>
#include <ATen/native/hammerblade/HammerBladeTensor.h>
#include <ATen/native/hammerblade/Offload.h>

#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>

namespace at { namespace native {
namespace {
void index_kernel_hb(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {

//  std::cout << "Enter index_kernel_hb function." << std::endl;
  auto strides = iter.get_strides();
  while (strides.size() < 2 * iter.ntensors()) {
    strides.push_back(0);
  }

  for(int i = 0; i < strides.size(); i++) {
    strides[i] = strides[i] / 4;
//    std::cout << "  strides[" << i << "] is: " << strides[i] << std::endl;
  }

  IntArrayRef kernel_strides = IntArrayRef(strides);
  IntArrayRef shapes = iter.shape();
  std::vector<eva_t> device_args;
  std::vector<eva_t> device_ptrs;
//  std::cout << "Number of tensors is " << iter.ntensors();
  for(int i = 0; i < iter.ntensors(); i++) {
    Tensor& t = iter.tensor(i);
    if(i > iter.noutputs() && t.dtype() == at::kLong) {
//      std::cout << "Convert data type" << std::endl;
      t.to(at::kInt);
    }
    device_args.push_back(create_device_tensor(t, device_ptrs));
//    std::cout << "  Tensor number of dimension is: " << t.ndimension() << std::endl;
//    std::cout << "  Tensor number of strides is: " << t.strides().size() << std::endl;
  }
  device_args.push_back(create_device_vector(shapes, true, device_ptrs));
  device_args.push_back(create_device_vector(kernel_strides, true, device_ptrs));
  device_args.push_back(create_device_vector(index_size, true, device_ptrs));
  device_args.push_back(create_device_vector(index_stride, true, device_ptrs));

  if(iter.ntensors() == 3) {
    c10::hammerblade::offload_kernel(
        "tensorlib_indexing_1d", device_args);
  } else if(iter.ntensors() == 4) {
    c10::hammerblade::offload_kernel(
        "tensorlib_indexing_2d", device_args);
  } else {
    AT_ERROR("Currently, we only support the indexing of 2D HB Tensor !");
  }
  
  cleanup_device(device_args, device_ptrs);
}

REGISTER_HAMMERBLADE_DISPATCH(index_stub, &index_kernel_hb);

}


// NOTE: Function updates addDim
Tensor collapseDims(Tensor t, int64_t &addDim) {
    // collapse dimensions smaller than addDim
    std::vector<int64_t> new_shape;
    if (addDim != 0) {
        int64_t new_size = 1;
        for (int i = 0; i < addDim; ++i) {
            new_size *= t.size(i);
        }
        new_shape.push_back(new_size);
    }

    // keep size of addDim
    new_shape.push_back(t.size(addDim));

    // collapse dimensions bigger than addDim
    if (addDim != t.dim() - 1) {
        int64_t new_size = 1;
        for (int i = addDim+1; i < t.dim(); ++i) {
            new_size *= t.size(i);
        }
        new_shape.push_back(new_size);
    }

    Tensor t_collapsed = t.view(IntArrayRef(new_shape));

    if (addDim != 0)
        addDim = 1;
    std::cout << "addDim: " << addDim << std::endl;
    std::cout << "new_shape: " << new_shape << std::endl;

    /*
    std::cout  << "addDim: " << addDim << std::endl;
    for (int i = 0; i < t_collapsed.dim(); ++i) {
        std::cout << "size[" << i << "]: " << t_collapsed.size(i) << std::endl;
        std::cout << "stride[" << i << "]: " << t_collapsed.stride(i) << std::endl;
    }
    */

    return t_collapsed;
}

Tensor& index_add_hb_(Tensor &self, int64_t dim, const Tensor &index, const Tensor &source) {
    auto  numel = index.numel();
    TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector");
    TORCH_CHECK(index.scalar_type() == ScalarType::Long, "index_add_(): Expected dtype int64 for index");
    TORCH_CHECK(self.scalar_type() == source.scalar_type(),
                "index_add_(): self and source must have the same scalar type");
    TORCH_CHECK(dim == 0 || dim < source.dim(),
                "index_add_(): Indexing dim ", dim, " is out of bounds of tensor");
    TORCH_CHECK(numel == (source.dim() == 0 ? 1 : source.size(dim)),
                "index_add_(): Number of indices should be equal to self.size(dim)");
    // TODO: check all dimension sizes besides dim dimension
    // TODO: check dim equal of self and source


    int64_t dst_add_dim = dim;
    int64_t src_add_dim = dim;
    // TODO: dst_add_dim and src_add_dim have the same size?
    Tensor dst_c = collapseDims(self, dst_add_dim);
    Tensor src_c = collapseDims(source, src_add_dim);


    // TODO: check sliceSize != 0
    // TODO: simplify sliceSize calculation?
    int64_t sliceSize = 1;
    for (int i = 0; i < dst_c.dim(); ++i) {
        if (i != dst_add_dim) {
            sliceSize *= dst_c.size(i);
        }
    }
    int64_t nbrIndices = index.numel();
    // TODO: Just last dim?
    bool indexShouldBeMajor = false;
    for (int i = 0; i < dst_c.dim(); ++i) {
        if (i != dst_add_dim && dst_c.size(i) > 1 && dst_c.stride(i) < dst_c.stride(dst_add_dim))
            indexShouldBeMajor = true;
    }
    if (indexShouldBeMajor)
        std::cout << "indexShouldBeMajor" << std::endl;
    else
        std::cout << "elementsInSliceMajor" << std::endl;

    std::vector<eva_t>  device_args;
    std::vector<eva_t>  device_ptrs;
    device_args.push_back(create_device_tensor(dst_c, device_ptrs));
    device_args.push_back(create_device_tensor(src_c, device_ptrs));
    device_args.push_back(create_device_tensor(index, device_ptrs));
    device_args.push_back(create_device_scalar((int64_t) dst_add_dim));
    device_args.push_back(create_device_scalar((int64_t) sliceSize));
    device_args.push_back(create_device_scalar((int64_t) nbrIndices));
    device_args.push_back(create_device_scalar((bool) indexShouldBeMajor));

    // TODO: numIndices <= 16
    // TODO: Implement small indices

    c10::hammerblade::offload_kernel("tensorlib_index_add_large_index", device_args);

    cleanup_device(device_args, device_ptrs);

    return self;
}

}}

