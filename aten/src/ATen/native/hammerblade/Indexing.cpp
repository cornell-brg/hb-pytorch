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


// NOTE: Function updates addDim to new addDim
// Function does not collapse dimensions of size 1 into addDim
// => dst and src tensors keep same number of dimensions
Tensor collapseDims(Tensor t, int &addDim) {
    // collapse dimensions smaller than addDim
    // int64_t necessary for IntArrayRef
    std::vector<int64_t> new_shape;
    if (addDim != 0) {
        int new_size = 1;
        for (int i = 0; i < addDim; ++i) {
            new_size *= t.size(i);
        }
        new_shape.push_back(new_size);
    }

    // keep size of addDim
    new_shape.push_back(t.size(addDim));

    // collapse dimensions bigger than addDim
    if (addDim != t.dim() - 1) {
        int new_size = 1;
        for (int i = addDim+1; i < t.dim(); ++i) {
            new_size *= t.size(i);
        }
        new_shape.push_back(new_size);
    }

    Tensor t_collapsed = t.view(IntArrayRef(new_shape));

    if (addDim != 0)
        addDim = 1;

    return t_collapsed;
}

Tensor& index_add_hb_(Tensor &self, int64_t dim, const Tensor &index_long, const Tensor &source) {
    TORCH_CHECK(index_long.scalar_type() == ScalarType::Long, "index_add_(): Expected dtype int64 for index input");

    Tensor index = index_long.toType(ScalarType::Int);
    TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector");
    TORCH_CHECK(index.scalar_type() == ScalarType::Int, "index_add_(): Index should have been converted to dtype int32");
    TORCH_CHECK(self.scalar_type() == source.scalar_type(),
                "index_add_(): self and source must have the same scalar type");
    TORCH_CHECK(dim == 0 || dim < source.dim(),
                "index_add_(): Indexing dim ", dim, " is out of bounds of tensor");
    TORCH_CHECK(index.numel() == (source.dim() == 0 ? 1 : source.size(dim)),
                "index_add_(): Number of indices should be equal to self.size(dim)");
    TORCH_CHECK(self.dim() == source.dim(), "index_add_(): Expected same number of dimensions for self and source tensors");
    for (int i = 0; i < self.dim(); ++i) {
        if (i != dim) {
            TORCH_CHECK(self.size(i) == source.size(i), "index_add_(): Expected equal size of dimension ", i, " for self and source tensors");
        }
    }
    // check if index tensor elements have valid values
    // index tensor needs to be duplicated on host-cpu to check elements on host
    Tensor index_cpu = index.cpu();
    auto index_elements = index_cpu.accessor<int32_t, 1>();
    for (int i = 0; i < index.numel(); i++) {
        int cur_idx = index_elements[i];
        TORCH_CHECK_INDEX((cur_idx >= 0) && (cur_idx < self.size(dim)), "index out of range in self");
    }


    int dst_add_dim = (int) dim;
    int src_add_dim = (int) dim;
    Tensor dst_c = collapseDims(self, dst_add_dim);
    Tensor src_c = collapseDims(source, src_add_dim);
    // Check dimensions of tensors after collapse; probably not necessary
    TORCH_CHECK(dst_c.dim() == src_c.dim(), "index_add_(): Expected same number of dimensions for dst_c and src_c tensors");
    TORCH_CHECK(dst_add_dim == src_add_dim, "index_add_(): Expected the generated Add-Index-Dimensions (dst_add_dim and src_add_dim) to be equal");
    for (int i = 0; i < dst_c.dim(); ++i) {
        if (i != dst_add_dim) {
            TORCH_CHECK(dst_c.size(i) == src_c.size(i), "index_add_(): Expected equal size of dimension ", i, " for dst_c and src_c tensors");
        }
    }

    int nbrIndices = index.numel();
    int sliceSize = dst_c.numel() / dst_c.size(dst_add_dim);
    TORCH_CHECK(sliceSize > 0, "index_add_(): Expected slice with size greater than 0");

    std::vector<eva_t>  device_args;
    std::vector<eva_t>  device_ptrs;
    device_args.push_back(create_device_tensor(dst_c, device_ptrs));
    device_args.push_back(create_device_tensor(src_c, device_ptrs));
    device_args.push_back(create_device_tensor(index, device_ptrs));
    device_args.push_back(create_device_scalar((int) dst_add_dim));
    device_args.push_back(create_device_scalar((int) sliceSize));

    if (nbrIndices <= 16) {
        // small number of indices
        c10::hammerblade::offload_kernel("tensorlib_index_add_small_index", device_args);
    } else {
        // large number of indices
        // indexMajorMode when addDim is not last dim, and not second-last (when last dim has size of 1)
        int indexMajorMode = 1;
        if ((dst_add_dim == dst_c.dim()-1) || (dst_c.size(dst_c.dim()-1) == 1 && dst_add_dim == dst_c.dim()-2)) {
            indexMajorMode = 0;
        }

        device_args.push_back(create_device_scalar((int) nbrIndices));
        device_args.push_back(create_device_scalar((int) indexMajorMode));
        c10::hammerblade::offload_kernel("tensorlib_index_add_large_index", device_args);
    }

    cleanup_device(device_args, device_ptrs);

    return self;
}

}}

