#include <ATen/native/sparse/SparseTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <TH/THBlasUtils.h>

namespace at { namespace native {

using namespace at::sparse;
// --------------------------------------------------------------------
// Utility functions
// --------------------------------------------------------------------

namespace {
  LongTensor _to_csr(const int64_t* indices, int64_t dim, int64_t nnz) {
    LongTensor csr = native::zeros({dim + 1}, kLong);

    // TODO: eliminate this conditional when zero-size dims supported correctly
    if (nnz > 0) {
      auto csr_accessor = csr.accessor<int64_t, 1>();
      // Convert the sparse matrix to CSR format
      at::parallel_for(0, nnz, 10000, [&](int64_t start, int64_t end) {
        int64_t h, hp0, hp1;
        for (auto i = start; i < end; i++) {
          hp0 = indices[i];
          hp1 = (i+1 == nnz) ?  dim : indices[i+1];
          if (hp0 != hp1) for (h = hp0; h < hp1; h++) {
            csr_accessor[h+1] = i+1;
          }
        }
      });
    }
    return csr;
  }

}

// --------------------------------------------------------------------
// zero_(SparseTensor)
// --------------------------------------------------------------------

// hummu hummu
SparseTensor& zero_sparse_(SparseTensor& self) {
  AT_ASSERT(self.is_sparse());
  at::zeros_out(self, get_sparse_impl(self)->sizes());
  return self._coalesced_(true);
}

// NB: Don't need zeros, zeros_like, already implemented in TensorFactories

// --------------------------------------------------------------------
// mul(SparseTensor, Scalar)
// --------------------------------------------------------------------

static Tensor wrapped_scalar_tensor(Scalar s) {
  auto tensor = scalar_to_tensor(s);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

SparseTensor& mul_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  if (is_same_tensor(r, t)) {
    r._values().mul_(value);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because mul_out takes Tensor&
    at::mul_out(r_values, t._values(), value);
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

SparseTensor& mul_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, Scalar value) {
  return mul_out_sparse_zerodim(r, t, wrapped_scalar_tensor(value));
}

// --------------------------------------------------------------------
// log1p(SparseTensor)
// --------------------------------------------------------------------

// TODO: add in-place variant

SparseTensor& log1p_out_sparse(SparseTensor& r, const SparseTensor& t) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());

  if (is_same_tensor(r, t)) {
    // don't have in-place log1p for uncoalesced input because coalesce() is not in-place
    TORCH_CHECK(
      r.is_coalesced(), "log1p: in-place on uncoalesced tensors is not supported yet!");
  }
  else {
    copy_sparse_to_sparse_(r, t.coalesce());
  }
  r._values().log1p_();
  return r;
}

SparseTensor& log1p_sparse_(SparseTensor& t) {
  TORCH_CHECK(t.is_coalesced(), "log1p: in-place on uncoalesced tensors is not supported yet!");
  return log1p_out_sparse(t, t);
}

// --------------------------------------------------------------------
// pow(SparseTensor, Scalar)
// --------------------------------------------------------------------

// TODO: add in-place variant

SparseTensor& pow_out_sparse_scalar(SparseTensor& r, const SparseTensor& t_, Scalar value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t_.is_sparse());
  TORCH_CHECK(value.toDouble() != 0, "pow: cannot raise to zeroth power on sparse tensor; it would make the result tensor dense");

  // This coalesce is why we can't easily provide an inplace variant
  SparseTensor t = t_.coalesce();

  r.resize_as_(t);
  auto indices = r._indices();
  indices.resize_as_(t._indices());
  indices.copy_(t._indices());
  Tensor r_values = r._values(); // Sigh... needed because pow_out takes Tensor&
  at::pow_out(r_values, t._values(), value);
  get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
  return r._coalesced_(t.is_coalesced());
}

SparseTensor pow_sparse_scalar(const SparseTensor& t, Scalar value) {
  SparseTensor r = at::empty({0}, t.options());
  pow_out_sparse_scalar(r, t, value);
  return r;
}

// --------------------------------------------------------------------
// div(SparseTensor, Scalar)
// --------------------------------------------------------------------

SparseTensor& div_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value);

Tensor div_sparse(const Tensor& self, const Tensor& value) {
  Tensor result = at::empty({0}, self.options());
  return div_out_sparse_zerodim(result, self, value);
}

Tensor& div_sparse_(Tensor& self, const Tensor& value) {
  return div_out_sparse_zerodim(self, self, value);
}

SparseTensor& div_out_sparse_zerodim(SparseTensor& r, const SparseTensor& t, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "sparse division only supports division by a scalar (got shape ",
      value.sizes(), " for argument 'other')");
  TORCH_CHECK(!value.is_sparse(), "A Sparse Tensor can only be divided by a scalar or zero-dim dense tensor");

  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());

  if (is_same_tensor(r, t)) {
    r._values().div_(value);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values(); // Sigh... needed because div_out takes Tensor&
    at::div_out(r_values, t._values(), value);
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

SparseTensor& div_out_sparse_scalar(SparseTensor& r, const SparseTensor& t, Scalar value) {
  return div_out_sparse_zerodim(r, t, wrapped_scalar_tensor(value));
}

// --------------------------------------------------------------------
// norm(SparseTensor, Scalar)
// --------------------------------------------------------------------

// Only supports floating point, FYI
Tensor norm_sparse(const SparseTensor& self, Scalar value) {
  AT_ASSERT(self.is_sparse());

  return self.coalesce()._values().norm(value);
}

// --------------------------------------------------------------------
// add(SparseTensor, SparseTensor, Scalar)  [broadcasts]
// --------------------------------------------------------------------

Tensor add_sparse(const Tensor& self, const Tensor& other, Scalar alpha) {
  // TODO: Why?! Can't we just flip the order here...
  TORCH_CHECK(!(self.is_sparse() && !other.is_sparse()),
              "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  if(self.is_hammerblade()) {
    Tensor result = at::empty(self.sizes(), self.options());
    return at::add_out(result, self, other, alpha);  
  }
  else {
    Tensor result = at::empty({0}, self.options());
    return at::add_out(result, self, other, alpha);  // redispatch!
  }
}

Tensor& add_sparse_(Tensor& self, const Tensor& other, Scalar alpha) {
  return at::add_out(self, self, other, alpha);  // redispatch!
}

// There's actually nothing sparse specific about these implementations

Tensor sub_sparse(const Tensor& self, const Tensor& other, Scalar alpha) {
  return native::add_sparse(self, other, -alpha);
}

Tensor& sub_sparse_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::add_sparse_(self, other, -alpha);
}

Tensor& sub_out_sparse(Tensor& r, const Tensor& self, const Tensor& other, Scalar alpha) {
  return at::add_out(r, self, other, -alpha);  // redispatch!
}

Tensor& add_out_dense_sparse_cpu(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, Scalar value);

SparseTensor& add_out_sparse_cpu(SparseTensor& r, const SparseTensor& t, const SparseTensor& src, Scalar value) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_cpu(r, t, src, value);
  }
  // TODO: This test seems a bit goofy
  TORCH_CHECK(src.is_sparse(), "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");
  AT_ASSERT(!t.is_cuda());  // the dispatch argument
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(t.sizes().equals(src.sizes()), "add: expected sizes of 'self' and 'other' to match, but ", t.sizes(), " != ", src.sizes());

  if (src._nnz() == 0) {
    return copy_sparse_to_sparse_(r, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r, src, value);
  }

  TORCH_CHECK(is_same_density(t, src), "add: expected 'self' and 'other' to have same density, but 'self' has ", t.sparse_dim(), " sparse dimensions while 'other' has ", src.sparse_dim(), " sparse dimensions");

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz(), max_nnz = t_nnz + s_nnz;
  bool t_coalesced = t.is_coalesced(), s_coalesced = src.is_coalesced();
  int64_t sparse_dim = src.sparse_dim();
  LongTensor t_indices = t._indices();
  Tensor t_values = t._values();
  LongTensor src_indices = src._indices();
  Tensor s_values = src._values();
  r.resize_as_(src);

  if (s_values.is_contiguous() && t_values.is_contiguous()) {
    LongTensor r_indices = at::empty({sparse_dim, max_nnz}, t_indices.options());
    Tensor r_values = new_values_with_size_of(s_values, max_nnz).zero_();
    get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);

    int64_t blockSize = r_values.stride(0);
    int64_t cmp, d;
    int64_t r_i = 0, t_i = 0, s_i = 0;

    // NB: relies on nnz tests above
    auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
    auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
    auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

    AT_DISPATCH_ALL_TYPES(
        t_values.scalar_type(), "cadd_sparse", [&] {
          scalar_t* t_values_ptr = t_values.data_ptr<scalar_t>();
          scalar_t* s_values_ptr = s_values.data_ptr<scalar_t>();
          scalar_t* r_values_ptr = r_values.data_ptr<scalar_t>();
          scalar_t cast_value = value.to<scalar_t>();
          while (t_i < t_nnz || s_i < s_nnz) {
            if (t_i >= t_nnz) {
              cmp = -1;
            } else if (s_i >= s_nnz) {
              cmp = 1;
            } else {
              cmp = 0;
              for (d = 0; d < sparse_dim; d++) {
                if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
                  cmp = 1;
                  break;
                }
                if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
                  cmp = -1;
                  break;
                }
              }
            }
            if (cmp >= 0) {
              for (d = 0; d < sparse_dim; d++) {
                r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
              }
              if (t_values.numel() > 0) {  // We add all elements from t_values to r_values only if t_values is not an empty tensor
                THBlas_axpy<scalar_t>(blockSize, 1,
                  t_values_ptr + t_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              t_i++;
            }
            if (cmp <= 0) {
              for (d = 0; d < sparse_dim; d++) {
                r_indices_accessor[d][r_i] = src_indices_accessor[d][s_i];
              }
              if (s_values.numel() > 0) {  // We add all elements from s_values to r_values only if s_values is not an empty tensor
                THBlas_axpy<scalar_t>(blockSize, cast_value,
                  s_values_ptr + s_i * blockSize, 1,
                  r_values_ptr + r_i * blockSize, 1);
              }
              s_i++;
            }
            r_i++;
          }
        }
    );

    get_sparse_impl(r)->set_nnz_and_narrow(r_i);
    // TODO: I think it may be possible to track inside the loop and
    // detect when we are uncoalesced (e.g., by observing that an
    // index goes backwards) which may be more precise than using the
    // coalesced flag here.  But this is easy.
    return r._coalesced_(t_coalesced && s_coalesced);
  } else {
    // If `t` or `src` contains non-contiguous `values`, `THBlas_axpy` doesn't work
    // and we concat the indices and values tensors instead.
    AT_DISPATCH_ALL_TYPES(
      s_values.scalar_type(), "add_out_sparse_cuda", [&] {
          if (value.to<scalar_t>() != static_cast<scalar_t>(1)) {
            s_values = s_values.mul(value);
          }
        });

    LongTensor r_indices = at::cat({t_indices, src_indices}, 1);
    Tensor r_values = at::cat({t_values, s_values}, 0);
    alias_into_sparse(r, r_indices, r_values);

    return r;
  }
}

// --------------------------------------------------------------------
// add(Tensor, SparseTensor, Scalar)
//    formerly known as spcadd
// --------------------------------------------------------------------

template <typename scalar_t>
void add_dense_sparse_worker_cpu(Tensor& r, Scalar value, const SparseTensor& sparse, const Tensor& indices, const Tensor& values) {
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto values_accessor = values.accessor<scalar_t, 1>();

  scalar_t* r_ptr = r.data_ptr<scalar_t>();
  scalar_t cast_value = value.to<scalar_t>();

  at::parallel_for(0, sparse._nnz(), 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++) {
      int64_t index = r.storage_offset();
      for (int64_t d = 0; d < sparse.sparse_dim(); d++) {
        index += r.stride(d) * indices_accessor[d][k];
      }
      r_ptr[index] += cast_value * values_accessor[k];
    }
  });
}

Tensor& add_out_dense_sparse_cpu(Tensor& r, const Tensor& dense, const SparseTensor& sparse_, Scalar value) {
  AT_ASSERT(!r.is_sparse());
  AT_ASSERT(!dense.is_sparse());
  AT_ASSERT(sparse_.is_sparse());

  AT_ASSERT(!dense.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "add: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "add: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(dense.sizes().equals(sparse_.sizes()), "add: expected 'self' and 'other' to have same size, but self has size ",
    dense.sizes(), " while other has size ", sparse_.sizes(), " (FYI: dense-sparse addition does not currently support broadcasting)");

  r.resize_as_(dense);
  SparseTensor sparse = sparse_.coalesce();

  LongTensor indices = sparse._indices();
  Tensor values = sparse._values();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse.sparse_dim();

  if (!is_same_tensor(r, dense)) r.copy_(dense);
  if (sparse._nnz() == 0) return r;

  // accessors rely on nnz test
  if (nDim > nDimI) {
    auto indices_accessor = indices.accessor<int64_t, 2>();
    for (int64_t k = 0; k < sparse._nnz(); k++) {
      Tensor dstBuffer = r;
      for (int64_t d = 0; d < sparse.sparse_dim(); d++) {
        dstBuffer = dstBuffer.select(0, indices_accessor[d][k]);
      }
      Tensor srcBuffer = values.select(0, k);
      dstBuffer.add_(srcBuffer, value);
    }
  } else {
    AT_DISPATCH_ALL_TYPES(
        values.scalar_type(), "add_dense_sparse", [&] {
          add_dense_sparse_worker_cpu<scalar_t>(r, value, sparse, indices, values);
        });
  }
  return r;
}

// --------------------------------------------------------------------
// mul(SparseTensor, SparseTensor)  [broadcasts]
// --------------------------------------------------------------------

Tensor mul_sparse(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return at::mul_out(result, self, other);  // redispatch!
}

Tensor& mul_sparse_(Tensor& self, const Tensor& other) {
  return at::mul_out(self, self, other);  // redispatch!
}

SparseTensor& mul_out_sparse_cpu(SparseTensor& r, const Tensor& t_, const Tensor& src_) {
  if (src_.dim() == 0) {
    return mul_out_sparse_zerodim(r, t_, src_);
  } else if (t_.dim() == 0) {
    return mul_out_sparse_zerodim(r, src_, t_);
  }

  TORCH_CHECK(t_.sizes().equals(src_.sizes()), "mul operands have incompatible sizes");
  AT_ASSERT(!t_.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "mul: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!src_.is_cuda(), "mul: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(t_.sizes().equals(src_.sizes()), "mul: expected 'self' and 'other' to have same sizes, but ", t_.sizes(), " != ", src_.sizes());

  if (src_._nnz() == 0 || t_._nnz() == 0) {
    r.resize_as_(src_);
    return r.zero_();
  }

  SparseTensor t = t_.coalesce();
  SparseTensor src = src_.coalesce();

  // saving those because they can be overwritten when doing in-place operations
  int64_t t_nnz = t._nnz(), s_nnz = src._nnz();
  int64_t max_nnz = std::min(t_nnz, s_nnz);  // multiply by zero is zero, and can be dropped
  int64_t sparse_dim = src.sparse_dim();
  LongTensor t_indices = t._indices();
  Tensor t_values = t._values();
  LongTensor src_indices = src._indices();
  Tensor s_values = src._values();
  LongTensor r_indices = at::empty({sparse_dim, max_nnz}, t_indices.options());
  Tensor r_values = new_values_with_size_of(t_values, max_nnz).zero_();
  r.resize_as_(src);
  get_sparse_impl(r)->set_indices_and_values_unsafe(r_indices, r_values);

  int64_t match, d;
  int64_t r_i = 0, t_i = 0, s_i = 0;

  // NB: relies on nnz test above
  auto t_indices_accessor = t_indices.accessor<int64_t, 2>();
  auto r_indices_accessor = r_indices.accessor<int64_t, 2>();
  auto src_indices_accessor = src_indices.accessor<int64_t, 2>();

  // Check if we can find matching indices, and if so, write an
  // entry to the result indices vector.  Returns true if matching
  // indices were found.
  auto index_preamble = [&]() {
    match = 1;
    for (d = 0; d < sparse_dim; d++) {
      if (t_indices_accessor[d][t_i] < src_indices_accessor[d][s_i]) {
        t_i++;
        match = 0;
        break;
      }
      if (t_indices_accessor[d][t_i] > src_indices_accessor[d][s_i]) {
        s_i++;
        match = 0;
        break;
      }
    }
    if (!match) return false;
    for (d = 0; d < sparse_dim; d++) {
      r_indices_accessor[d][r_i] = t_indices_accessor[d][t_i];
    }
    return true;
  };

  if (t_values.dim() > 1) {
    while (t_i < t_nnz && s_i < s_nnz) {
      if (!index_preamble()) continue;
      r_values.select(0, r_i).addcmul_(t_values.select(0, t_i), s_values.select(0, s_i));
      r_i++;
      t_i++;
      s_i++;
    }
  } else {
    AT_DISPATCH_ALL_TYPES(
        r_values.scalar_type(), "mul_out_sparse", [&] {
          auto r_accessor = r_values.accessor<scalar_t, 1>();
          auto t_accessor = t_values.accessor<scalar_t, 1>();
          auto s_accessor = s_values.accessor<scalar_t, 1>();

          while (t_i < t_nnz && s_i < s_nnz) {
            if (!index_preamble()) continue;
            r_accessor[r_i] = t_accessor[t_i] * s_accessor[s_i];
            r_i++;
            t_i++;
            s_i++;
          }
        }
    );
  }

  get_sparse_impl(r)->set_nnz_and_narrow(r_i);
  return r._coalesced_(true);
}

// --------------------------------------------------------------------
// addmm(D1, S, D2, beta, alpha) -> D  [broadcasts]
//
// D = beta * D1 + alpha * mm(S, D2)
// --------------------------------------------------------------------

template <typename scalar_t>
void s_addmm_out_sparse_dense_worker(int64_t nnz, int64_t dim_i, int64_t dim_j, int64_t dim_k, Tensor& r, Scalar beta, const Tensor& t, Scalar alpha, const Tensor& indices, const Tensor& values, const Tensor& dense) {
  int64_t i;

  // r_ = alpha * sparse * dense
  scalar_t cast_alpha = alpha.to<scalar_t>();
  scalar_t cast_beta = beta.to<scalar_t>();
  if (cast_beta == 0) {
    r.zero_();
  } else if (cast_beta == 1) {
    if (!is_same_tensor(r, t)) {
      r.copy_(t);
    }
  } else {
    at::mul_out(r, t, scalar_to_tensor(beta));
  }

  auto indices_accessor = indices.accessor<int64_t, 2>();

  auto values_accessor = values.accessor<scalar_t, 1>();
  scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
  scalar_t* r_ptr = r.data_ptr<scalar_t>();

  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t r_stride0 = r.stride(0);
  int64_t r_stride1 = r.stride(1);
  for (i = 0; i < nnz; i++) {
    scalar_t val = values_accessor[i];
    int64_t row = indices_accessor[0][i];
    int64_t col = indices_accessor[1][i];
    if (col >= 0 && col < dim_j && row >= 0 && row < dim_i) {
      THBlas_axpy<scalar_t>(dim_k,
            cast_alpha * val,
            dense_ptr + col * dense_stride0, dense_stride1,
            r_ptr + row * r_stride0, r_stride1);
    } else {
      if (col < 0 || col >= dim_j) {
        AT_ERROR("addmm: index out of column bound: ", col, " not between 1 and ", dim_j);
      } else {
        AT_ERROR("addmm: index out of row bound: ", row, " not between 1 and ", dim_i);
      }
    }
  }
};

Tensor& s_addmm_out_sparse_dense_cpu(
    Tensor& r,
    const Tensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  // TODO: This error message seems awfully opaque
  AT_ASSERT(!t.is_cuda());
  TORCH_CHECK(!r.is_cuda(), "addmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "addmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(!dense.is_cuda(), "addmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0, "addmm: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2, "addmm: matrices expected, got ", dense.dim(), "D tensor");

  // ixj * jxk = ixk
  int64_t dim_i = sparse_.size(0);
  int64_t dim_j = sparse_.size(1);
  int64_t dim_k = dense.size(1);

  TORCH_CHECK(dense.size(0) == dim_j,
      "addmm: Argument #3 (dense): Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  TORCH_CHECK(t.size(0) == dim_i,
      "addmm: Argument #1 (t): Expected dim 0 size ", dim_i, ", got ", t.size(0));
  TORCH_CHECK(t.size(1) == dim_k,
      "addmm: Argument #1 (t): Expected dim 1 size ", dim_k, ", got ", t.size(1));

  r.resize_({dim_i, dim_k});

  int64_t nnz        = sparse_._nnz();

  if (nnz == 0) {
    at::mul_out(r, t, at::scalar_tensor(beta, r.options()));
    return r;
  }

  LongTensor indices = sparse_._indices();
  Tensor values      = sparse_._values();

  AT_DISPATCH_ALL_TYPES(
      values.scalar_type(), "addmm_sparse_dense", [&] {
        s_addmm_out_sparse_dense_worker<scalar_t>(nnz, dim_i, dim_j, dim_k, r, beta, t, alpha, indices, values, dense);
      }
  );

  return r;

}

Tensor& addmm_out_sparse_dense_cpu(
    Tensor& result,
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha
) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_dense_cpu(result, b_self, mat1, mat2, beta, alpha);
}

Tensor s_addmm_sparse_dense_cpu(
    const Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  Tensor r = at::empty({0}, t.options());
  s_addmm_out_sparse_dense_cpu(r, t, sparse, dense, beta, alpha);
  return r;
}

Tensor addmm_sparse_dense_cpu(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha
) {
  Tensor b_self;
  std::tie(b_self) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_sparse_dense_cpu(b_self, mat1, mat2, beta, alpha);
}

Tensor& s_addmm_sparse_dense_cpu_(
    Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  return s_addmm_out_sparse_dense_cpu(t, t, sparse, dense, beta, alpha);
}

// NB: Purposely no broadcasting version of addmm inplace

Tensor _sparse_addmm(
  const Tensor& t,
  const SparseTensor& sparse,
  const Tensor& dense,
  Scalar beta,
  Scalar alpha
) {
  // _sparse_addmm forward is functionally equivalent to addmm; it's
  // just the backward that is different.  This technically does an
  // unnecessary redispatch, I was too lazy to make it not do that
  return at::addmm(t, sparse, dense, beta, alpha);
}

Tensor _sparse_mm(
  const SparseTensor& sparse,
  const Tensor& dense
) {
  Tensor t = at::zeros({}, dense.options());
  return at::_sparse_addmm(t, sparse, dense, 0, 1);  // redispatch!
}

// NB: Despite its suggestive name, this actually only exists so that
// we can redispatch to addmm_out; this is NOT an implementation of
// the sparse masking version of mm
SparseTensor& _sparse_mm_out(
  SparseTensor& result,
  const SparseTensor& sparse,
  const Tensor& dense
) {
  Tensor t = at::zeros({}, dense.options());
  return at::addmm_out(result, t, sparse, dense, 0, 1);  // redispatch!
}

// --------------------------------------------------------------------
// hspmm(SparseTensor mat1, Tensor mat2)
// --------------------------------------------------------------------

SparseTensor& hspmm_out_sparse_cpu(SparseTensor& r, const SparseTensor& sparse_, const Tensor& dense) {
  // TODO: Make this a real argument
  Scalar alpha = 1;

  AT_ASSERT(!sparse_.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "hspmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!dense.is_cuda(), "hspmm: expected 'other' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "hspmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "hspmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2,
      "hspmm: Argument #3: matrices expected, got ", dense.dim(), "D tensor");

  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  TORCH_CHECK(dense.size(0) == k,
      "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

  get_sparse_impl(r)->raw_resize_(1, 1, {m, n});

  SparseTensor sparse = sparse_.coalesce();

  int64_t nnz = sparse._nnz();

  if (nnz == 0) {
    r.zero_();
    return r;
  }

  LongTensor indices = at::empty({1, nnz}, at::initialTensorOptions().dtype(kLong));

  // Initialize the sparse matrix that will be used with spaddmm to send rows
  // from the dense matrix to rows of the output's value tensor
  SparseTensor newSparse = sparse.clone();
  LongTensor spIndices = newSparse._indices();
  LongTensor valueIndices = spIndices.select(0, 0);

  // Compute output indices
  auto valueIndices_accessor = valueIndices.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();

  int64_t i = -1, prevIdx = -1;
  for (int64_t j = 0; j < nnz; j++) {
    int64_t currIdx = valueIndices_accessor[j];
    if (currIdx != prevIdx) {
      indices_accessor[0][++i] = currIdx;
      prevIdx = currIdx;
    }
    valueIndices_accessor[j] = i;
  }
  int64_t outNnz = i + 1;
  indices.resize_({1, outNnz});
  Tensor values = at::empty({outNnz, n}, dense.options());

  std::vector<int64_t> new_size = get_sparse_impl(newSparse)->sizes().vec();
  new_size[0] = outNnz;
  get_sparse_impl(newSparse)->raw_resize_(get_sparse_impl(newSparse)->sparse_dim(), get_sparse_impl(newSparse)->dense_dim(), new_size);

  // Compute output values tensor with sparse * dense multiplication
  s_addmm_out_sparse_dense_cpu(values, values, newSparse, dense, 0, alpha);
  get_sparse_impl(r)->set_indices_and_values_unsafe(indices, values);

  return r;
}

SparseTensor hspmm_sparse_cpu(const SparseTensor& sparse, const Tensor& dense) {
  SparseTensor r = at::empty({0}, sparse.options());
  hspmm_out_sparse_cpu(r, sparse, dense);
  return r;
}

// --------------------------------------------------------------------
// sspaddmm(S1, S2, D, beta, alpha) -> S
//
// S = beta * S1 + alpha * mm(S2, D)
// --------------------------------------------------------------------

SparseTensor& _sspaddmm_out_cpu(
    SparseTensor& r,
    const SparseTensor& t,
    const SparseTensor& sparse_,
    const Tensor& dense,
    Scalar beta,
    Scalar alpha
) {
  AT_ASSERT(!t.is_cuda()); // dispatch argument
  TORCH_CHECK(!r.is_cuda(), "sspaddmm: expected 'out' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!sparse_.is_cuda(), "sspaddmm: expected 'mat1' to be a CPU tensor, but got a CUDA tensor");
  TORCH_CHECK(!dense.is_cuda(), "sspaddmm: expected 'mat2' to be a CPU tensor, but got a CUDA tensor");

  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "sspaddmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2,
      "sspaddmm: Argument #2: matrices expected, got ", dense.dim(), "D tensor");

  SparseTensor sparse = sparse_.coalesce();

  // ixj * jxk = ixk
  int64_t dim_i = sparse.size(0);
  int64_t dim_j = sparse.size(1);
  int64_t dim_k = dense.size(1);

  // NB: This has to occur before the checks, because r may alias t.
  // See test_saddmm
  get_sparse_impl(r)->raw_resize_(2, 0, {dim_i, dim_k});

  TORCH_CHECK(dense.size(0) == dim_j,
      "sspaddmm: Argument #3: Expected dim 0 size ", dim_j, ", got ", dense.size(0));
  TORCH_CHECK(t.size(0) == dim_i,
      "sspaddmm: Argument #1: Expected dim 0 size ", dim_i, ", got ", t.size(0));
  TORCH_CHECK(t.size(1) == dim_k,
      "sspaddmm: Argument #1: Expected dim 1 size ", dim_k, ", got ", t.size(1));

  int64_t nnz        = sparse._nnz();
  LongTensor indices = sparse._indices();
  Tensor values      = sparse._values();

  LongTensor csr = _to_csr(indices.data_ptr<int64_t>(), dim_i, nnz);

  int64_t t_nnz = t._nnz();
  int64_t r_nnz = nnz * dim_k + t_nnz;
  LongTensor newi = at::empty({2, r_nnz}, kLong);
  LongTensor newv = native::zeros({r_nnz}, values.options());

  if (t_nnz != 0) {
    LongTensor narrowi = newi.narrow(1, 0, t_nnz);
    Tensor narrowv = newv.narrow(0, 0, t_nnz);

    narrowi.copy_(t._indices());
    narrowv.copy_(t._values());
    newv.mul_(beta);
  }

  // sparse = sparse * dense
  int64_t p = t_nnz;

  auto csr_accessor = csr.accessor<int64_t, 1>();
  auto indices_accessor = indices.accessor<int64_t, 2>();
  auto newi_accessor = newi.accessor<int64_t, 2>();

  int64_t dense_stride0 = dense.stride(0);
  int64_t dense_stride1 = dense.stride(1);
  int64_t newv_stride0 = newv.stride(0);

  AT_DISPATCH_ALL_TYPES(
      values.scalar_type(), "sspmm", [&] {
        auto values_accessor = values.accessor<scalar_t, 1>();
        scalar_t* dense_ptr = dense.data_ptr<scalar_t>();
        scalar_t* newv_ptr = newv.data_ptr<scalar_t>();
        scalar_t cast_alpha = alpha.to<scalar_t>();

        for (int64_t h = 0; h < dim_i; h++) {
          int64_t i_start = csr_accessor[h];
          int64_t i_end = csr_accessor[h+1];
          for (int64_t i = i_start; i < i_end; i++) {
            scalar_t val = values_accessor[i];
            int64_t col = indices_accessor[1][i];
            if (col >= 0 && col < dim_j) {
              THBlas_axpy<scalar_t>(dim_k,
                  cast_alpha * val,
                  dense_ptr + col * dense_stride0, dense_stride1,
                  newv_ptr + p * newv_stride0, 1);
            } else {
              AT_ERROR("index out of bound. sspmm: ", col, " not between 1 and ", dim_j);
            }
          }
          // Fill up the indices with the right values
          if (i_start != i_end) {
            for (int64_t i = 0; i < dim_k; i++) {
              newi_accessor[0][p+i] = h;
              newi_accessor[1][p+i] = i;
            }
            p += dim_k;
          }
        }
      }
  );

  // to avoid a clone
  get_sparse_impl(r)->set_indices_and_values_unsafe(newi, newv);
  get_sparse_impl(r)->set_nnz_and_narrow(p);

  return r;
}

// sparse, sparse, sparse, dense, real, real -> sparse
Tensor& _sspaddmm_out_only_sparse(Tensor& result, const Tensor& self,
    const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  AT_ERROR("tensor.sspaddmm(...) can only be called on sparse tensors");
}

// sparse, dense -> sparse
Tensor smm(const Tensor& self, const Tensor& mat2) {
  auto result = at::empty({0}, self.options());
  at::sspaddmm_out(result, result, self, mat2, 0.0, 1.0);
  return result;
}

// sparse, sparse, dense, real, real -> sparse
Tensor sspaddmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta, Scalar alpha) {
  auto result = at::empty({0}, self.options());
  at::sspaddmm_out(result, self, mat1, mat2, beta, alpha);
  return result;
}

// --------------------------------------------------------------------
// sparse.sum()
//
// This implementation calls coalesce() to do the sum reduction on
// sparse dims. Ideally in the future there should be unified reduction function
// for ops like sum, max, and min.
// --------------------------------------------------------------------
Tensor _sparse_sum(const SparseTensor& input) {
  return input.coalesce().values().sum();
}

Tensor _sparse_sum(const SparseTensor& input, ScalarType dtype) {
  // don't have to do a conversion to the correct dtype first
  // just need to setup the accumulator correctly
  return input.coalesce().values().sum(dtype);
}

Tensor _sparse_sum(const SparseTensor& input, IntArrayRef dims_to_sum, ScalarType dtype) {
  return at::_sparse_sum(input.to(dtype), dims_to_sum);
}

Tensor _sparse_sum(const SparseTensor& input, IntArrayRef dims_to_sum) {
  TORCH_CHECK(input._nnz() > 0, "_sparse_sum: sparse tensor input._nnz() == 0, please call torch.sparse.sum(input) instead.")

  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  LongTensor indices = input._indices();
  Tensor values = input._values();
  IntArrayRef sizes = input.sizes();
  const int64_t sparse_dim = input.sparse_dim();
  // const int64_t dense_dim = input.dense_dim();

  auto dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (int64_t d = 0; d < input_dim; d++) {
    if (dims_to_sum_b[d]) {
      if (d >= sparse_dim) dense_dims_to_sum_v.emplace_back(d + 1 - sparse_dim);
    }
    else {
      dims_to_keep_v.emplace_back(d);
    }
  }
  const int64_t sparse_dims_to_sum_size = dims_to_sum_v.size() - dense_dims_to_sum_v.size();
  const bool sum_all_sparse_dim = (sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);

  // new values
  Tensor new_values;
  if (sum_dense_dim) {
    new_values = values.sum(dense_dims_to_sum_v);
  }
  else {
    new_values = values.clone(at::MemoryFormat::Contiguous);
  }

  if (sum_all_sparse_dim) {
    // return a dense tensor if sum over all sparse dims
    new_values = new_values.sum(0);
    return new_values;
  }
  else { // !sum_all_sparse_dim
    // new indices
    LongTensor new_indices;
    if (sparse_dims_to_sum_size == 0) {
      new_indices = indices.clone(at::MemoryFormat::Contiguous);
    }
    else {
      new_indices = at::empty({sparse_dim - sparse_dims_to_sum_size, input._nnz()}, indices.options());
      for (int64_t i = 0; i < dims_to_keep_v.size(); i++) {
        int64_t d = dims_to_keep_v[i];
        if (d < sparse_dim) new_indices[i].copy_(indices[d]);
        else break;
      }
    }

    // new size
    int64_t new_sparse_dim = new_indices.size(0);
    int64_t new_dense_dim = new_values.dim() - 1; // exclude nnz dim
    std::vector<int64_t> new_sizes;
    for (auto d : dims_to_keep_v) new_sizes.emplace_back(sizes[d]);
    if (sum_all_sparse_dim) new_sizes.emplace(new_sizes.begin(), 1);

    // use coalesce() to do sum reduction
    SparseTensor new_sparse = at::_sparse_coo_tensor_with_dims_and_tensors(new_sparse_dim, new_dense_dim, new_sizes, new_indices, new_values, input.options());
    new_sparse = new_sparse.coalesce();
    return new_sparse;
  }

}

// --------------------------------------------------------------------
// NOTE [ sparse.sum() backward ]
//
// When sum over sparse_dim, backward scatters gradients from grad tensor to input tensor.
// Grad and input need to align indices over sparse_dim that are not summed (given
// input.spares_dim >= grad.sparse_dim). Implementation here compares each pair of
// indices between grad and input. When a matching indices pair (input_i, grad_i) is found,
// copy grad.values[grad_i] -> input_grad.values[input_i]. E.g.,
//
//  input.sparse_dim = [5, 5]
//  input.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
//                   [1, 4, 4, 0, 1, 3, 2, 4]]
//  input.values =   [0, 1, 2, 3, 4, 5, 6, 7]
//  ...
//  sparse.sum(input, [0])
//  backward(...)
//  ...
//  grad.indices = [[0, 1, 2, 3]]
//  grad.values =   [1, 2, 0, 4]
//
// # after indices matching
//         input         grad
//        [[0, 1],   ->  [1]
//         [0, 4],   ->  [ ]
//         [1, 4],   ->  [ ]
//         [2, 0],   ->  [0]
//         [2, 1],   ->  [1]
//         [3, 3],   ->  [3]
//         [4, 2],   ->  [2]
//         [4, 4]])  ->  [ ]
//
// input_grad.indices = [[0, 0, 1, 2, 2, 3, 4, 4],
//                       [1, 4, 4, 0, 1, 3, 2, 4]]
// input_grad.values =   [2, 0, 0, 1, 2, 4, 0, 0]
//
// Note that we allow input to be uncoalesced in the forward,
// we have to coalesce input at the backward, because grad-of-input
// take the same indices as input, if input is not coalesced, then
// coalescing grad-of-input may add up grad values for a duplicate indices,
// and hence generates a wrong grad-of-input.
//
// Other edge cases:
// - assign zero values to input gradients if cannot find matched indices at grad
// - grad.values might have zeros
// --------------------------------------------------------------------
Tensor _sparse_sum_backward_cpu(const Tensor& grad_, const SparseTensor& input_, IntArrayRef dims_to_sum) {
  TORCH_CHECK(!grad_.is_cuda(), "_sparse_sum_backward_cpu: expected 'grad_' to be CPU tensor, but got CUDA tensor");
  TORCH_CHECK(!input_.is_cuda(), "_sparse_sum_backward_cpu: expected 'input_' to be CPU tensor, but got CUDA tensor");

  auto input = input_.coalesce();
  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  LongTensor input_indices = input._indices();
  Tensor input_values = input._values();
  IntArrayRef input_sizes = input.sizes();
  const int64_t input_sparse_dim = input.sparse_dim();
  const int64_t input_dense_dim = input.dense_dim();
  const int64_t input_nnz = input._nnz();

  int64_t sparse_dims_to_sum_size = 0;
  auto sparse_dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (int64_t d = 0; d < input_dim; d++) {
    if (dims_to_sum_b[d]) {
      if (d < input_sparse_dim) sparse_dims_to_sum_size ++;
      else dense_dims_to_sum_v.emplace_back(d + 1 - input_sparse_dim);
    }
    else {
      if (d < input_sparse_dim) sparse_dims_to_keep_v.emplace_back(d);
    }
  }

  const bool sum_all_sparse_dim = (input_sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);
  const bool sum_sparse_dim = (sparse_dims_to_sum_size > 0);

  if (sum_all_sparse_dim) {
    TORCH_CHECK(!grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be dense since all sparse dims are summed");
    auto grad_input_values = grad_;
    auto expand_size = input_values.sizes().vec();
    if (sum_dense_dim) {
      auto dense_expand_size = std::vector<int64_t>(expand_size);
      dense_expand_size.erase(dense_expand_size.begin());
      AT_ASSERT(dense_expand_size.size() == (input_values.dim() - 1));
      for (auto d : dense_dims_to_sum_v) grad_input_values = grad_input_values.unsqueeze(d - 1);  // -1 since grad has no nnz dim
      grad_input_values = grad_input_values.expand(dense_expand_size);
    }
    grad_input_values = grad_input_values.expand(expand_size).clone(at::MemoryFormat::Contiguous);
    return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(at::MemoryFormat::Contiguous), grad_input_values, input.options().dtype(grad_.dtype())); // convert to grad dtype
  }
  else {
    TORCH_CHECK(grad_.is_sparse(), "_sparse_sum_backward_cpu: expected grad_ Tensor to be sparse, but got dense");
    auto grad = grad_.coalesce();
    LongTensor grad_indices = grad._indices();
    Tensor grad_values = grad._values();
    const int64_t grad_sparse_dim = grad.sparse_dim();
    const int64_t grad_nnz = grad._nnz();

    Tensor grad_values_expand = grad_values;
    if (sum_dense_dim) {
      auto expand_size = input_values.sizes().vec();
      if (sum_sparse_dim) expand_size[0] = grad_values.size(0);
      for (auto d : dense_dims_to_sum_v) grad_values_expand = grad_values_expand.unsqueeze(d);
      grad_values_expand = grad_values_expand.expand(expand_size).clone(at::MemoryFormat::Contiguous);
    }

    Tensor grad_input_values;
    if (sum_sparse_dim) {
      // see NOTE [ sparse.sum() backward ]
      grad_input_values = at::zeros_like(input_values, grad_values.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      // get flatten indices for grad and input
      auto grad_sparse_dim_to_keep_v = std::vector<int64_t>(grad_sparse_dim);
      std::iota(grad_sparse_dim_to_keep_v.begin(), grad_sparse_dim_to_keep_v.end(), 0);

      auto grad_indices_1D = flatten_indices_by_dims(grad_indices, grad.sizes(), grad_sparse_dim_to_keep_v); // flatten indices on all sparse_dim of grad, output indices is coalesced and sorted
      auto grad_indices_1D_accessor = grad_indices_1D.accessor<int64_t, 1>();
      auto input_indices_1D = flatten_indices_by_dims(input_indices, input_sizes, sparse_dims_to_keep_v);
      auto input_indices_1D_accessor = input_indices_1D.accessor<int64_t, 1>();

      // binary search to find matching indices

      at::parallel_for(0, input_nnz, 0, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
          int64_t input_idx = input_indices_1D_accessor[i];
          int64_t l = 0, r = grad_nnz - 1;
          while (l <= r) {
            int64_t m = l + (r - l) / 2;
            if (grad_indices_1D_accessor[m] == input_idx) {
              grad_input_values[i].copy_(grad_values_expand[m]);
              break;
            }
            if (grad_indices_1D_accessor[m] < input_idx) {
              l = m + 1;
            }
            else {
              r = m - 1;
            }
          }
        }
      });
    }
    else {
      grad_input_values = grad_values_expand;
    }
    return at::_sparse_coo_tensor_with_dims_and_tensors(input_sparse_dim, input_dense_dim, input_sizes, input_indices.clone(at::MemoryFormat::Contiguous), grad_input_values, grad.options());
  }
}

constexpr int cpu_dense_input_batch_size_dim = 0;
constexpr int cpu_dense_input_channels_dim = 1;
constexpr int cpu_sparse_weight_channels_dim = 0;
constexpr int cpu_dense_output_channels_dim = 1;

static std::vector<int64_t> cpu_sparse_conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[cpu_dense_input_batch_size_dim];
  output_size[1] = weight_size[cpu_sparse_weight_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

Tensor sparse_convolution(const Tensor& input_t, const SparseTensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef stride,
    IntArrayRef dilation, int64_t groups) {

  auto output_t = at::zeros(
                    cpu_sparse_conv_output_size(input_t.sizes(), weight_t.sizes(),
                                     padding, stride, dilation, groups),
                    input_t.options());
  /*
  Sparse Convolution implementation for CPU.
  */
  return output_t;
}

Tensor mv_sparse(const SparseTensor& self, const Tensor& vec)
{
  TORCH_CHECK(self.ndimension() == 2 && 
              vec.ndimension() == 1,
              "mv: two tensor dim should be 2 and 1, but got ",
              "SparseTensor Dim: ", self.ndimension(), "Tensor Dim: ", vec.ndimension());

  TORCH_CHECK(vec.size(-1) == self.size(-1),
              "mv: expected self.size(-1) == vec.size(-1)");

  auto result = self.matmul(vec.unsqueeze(-1));

  return result.squeeze(-1);
}

// --------------------------------------------------------------------
// Sinkhorn Functions
// --------------------------------------------------------------------

// --------------------------------------------------------------------
// sddtmm(SparseTensor, Tensor, Tensor) -> SparseTensor
//
// sddtmm(a, b, c) returns (b@c.T) at locations were a is nonzero
// --------------------------------------------------------------------

template <typename scalar_t>
void sddtmm_kernel_cpu(
  const SparseTensor& a_sparse_tensor,
  const Tensor& b_dense_tensor,
  const Tensor& c_dense_tensor,
  Tensor& out_indices,
  Tensor& out_vals){
    
  if ( b_dense_tensor.scalar_type() != ScalarType::Float
    || c_dense_tensor.scalar_type() != ScalarType::Float ) {
    AT_ERROR("SddTmm is implemented for Float type only for matrices b and c"); 
  }

  TORCH_CHECK(a_sparse_tensor.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'a' in sddtmm!");
  TORCH_CHECK(b_dense_tensor.dim() == 2 && c_dense_tensor.dim() == 2, "Expected 2D matrixes for 'a' and 'b', but got ", b_dense_tensor.dim(), " and ", c_dense_tensor.dim(), " tensors");
  TORCH_CHECK(b_dense_tensor.size(1) == c_dense_tensor.size(1), "Matrix multiply dimension mismatch: 'b' dim 1 = ", b_dense_tensor.size(1), ", 'c'.T dim 0 = ", c_dense_tensor.size(1));

  TORCH_CHECK(b_dense_tensor.size(0) == a_sparse_tensor.size(0) && c_dense_tensor.size(0) == a_sparse_tensor.size(1),"SddTmm sample dimension mismatch: sample was shape ",a_sparse_tensor.size(0)," by ",a_sparse_tensor.size(1),", but b@c is shape ",b_dense_tensor.size(0)," by ",c_dense_tensor.size(0));

  auto indices = a_sparse_tensor._indices();
  TORCH_CHECK(indices.dtype() == at::kLong, "Indices should be long, but got ", indices.dtype());

  auto a_indices = indices.accessor<int64_t, 2>();
  auto b_dense = b_dense_tensor.accessor<scalar_t, 2>();
  auto c_dense = c_dense_tensor.accessor<scalar_t, 2>();
  auto res_indices = out_indices.accessor<int64_t, 2>();
  auto res_vals = out_vals.accessor<scalar_t, 1>();

  int dot_len = b_dense.size(1);
  for (int k = 0; k < a_sparse_tensor._nnz(); k++) {
    int row = a_indices[0][k]; //0
    int col = a_indices[1][k];  //1

    float dot_total = 0; 
    for (int i = 0; i < dot_len; i++){
      dot_total += b_dense[row][i] * c_dense[col][i];
    }
    
    // update indices & vals
    res_indices[0][k] = row;
    res_indices[1][k] = col;
    res_vals[k] = dot_total;
  }
}

// --------------------------------------------------------------------
// sddtmm(Csr Tensor, Tensor, Tensor) -> SparseTensor
//
// sddtmm(a, b, c) returns (b@c.T) at locations were a is nonzero
// optimized version from above since it uses the CSR format for the sparse matrix and allows more reuse!
// --------------------------------------------------------------------
template <typename scalar_t>
void sddtmm_kernel_cpu_opt(
  const LongTensor& a_csr_tensor,
  const LongTensor& a_cols_tensor,
  const Tensor& b_dense_tensor,
  const Tensor& c_dense_tensor,
  Tensor& out_indices,
  Tensor& out_vals){
    
  if ( b_dense_tensor.scalar_type() != ScalarType::Float
    || c_dense_tensor.scalar_type() != ScalarType::Float ) {
    AT_ERROR("SddTmm is implemented for Float type only for matrices b and c"); 
  }

  TORCH_CHECK(b_dense_tensor.dim() == 2 && c_dense_tensor.dim() == 2, "Expected 2D matrixes for 'a' and 'b', but got ", b_dense_tensor.dim(), " and ", c_dense_tensor.dim(), " tensors");
  TORCH_CHECK(b_dense_tensor.size(1) == c_dense_tensor.size(1), "Matrix multiply dimension mismatch: 'b' dim 1 = ", b_dense_tensor.size(1), ", 'c'.T dim 0 = ", c_dense_tensor.size(1));

  auto a_csr = a_csr_tensor.accessor<int64_t, 1>();
  auto a_cols = a_cols_tensor.accessor<int64_t, 1>();

  auto b_dense = b_dense_tensor.accessor<scalar_t, 2>();
  auto c_dense = c_dense_tensor.accessor<scalar_t, 2>();
  auto res_indices = out_indices.accessor<int64_t, 2>();
  auto res_vals = out_vals.accessor<scalar_t, 1>();

  int dot_len = b_dense.size(1);

  int out_row = b_dense.size(0);
  int out_cols = c_dense.size(0);

  int nnz_idx=0;
  // int flops = 0;
  for (int a_row = 0; a_row < out_row; a_row++) {
    for (int a_col_idx = a_csr[a_row]; a_col_idx < a_csr[a_row+1]; a_col_idx++){
      int a_col = a_cols[a_col_idx];

      float dot_total = 0; 
      // auto res_dot = b_dense_tensor.select(0,a_row).dot(c_dense_tensor.select(0,a_col));
      // dot_total = *res_dot.data_ptr<scalar_t>();
      for (int i = 0; i < dot_len; i++){
        dot_total += b_dense[a_row][i] * c_dense[a_col][i];
        // flops+=2;
      }
      // update indices & vals
      res_indices[0][nnz_idx] = a_row;
      res_indices[1][nnz_idx] = a_col;
      res_vals[nnz_idx] = dot_total;
      ++nnz_idx;
    }
  }
  // printf("FLOPS for sddtmm %d\n",flops);
}


SparseTensor sddtmm_cpu(
  const SparseTensor& a_sparse_tensor,
  const Tensor& b_dense_tensor,
  const Tensor& c_dense_tensor) {
  
  Tensor result_indices = at::empty({2, a_sparse_tensor._nnz()}, {at::requires_grad().device(at::kCPU).dtype(at::kLong)});
  Tensor result_vals = at::empty(a_sparse_tensor._nnz(), {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});

  LongTensor indices = a_sparse_tensor._indices();
  LongTensor a_colIndices = indices.select(0, 1);
  int64_t* a_rowIndices = indices.select(0, 0).data_ptr<int64_t>();
  int64_t a_nnz = a_sparse_tensor._nnz();
  int64_t a_dim = a_sparse_tensor.size(0);
  LongTensor a_csr = _to_csr(a_rowIndices, a_dim, a_nnz);

  // AT_DISPATCH_ALL_TYPES(b_dense_tensor.scalar_type(), "sddtmm_cpu", [&]{
  //   sddtmm_kernel_cpu<scalar_t>(a_sparse_tensor, b_dense_tensor, c_dense_tensor, result_indices, result_vals);
  // });
  
  AT_DISPATCH_ALL_TYPES(b_dense_tensor.scalar_type(), "sddtmm_cpu", [&]{
    sddtmm_kernel_cpu_opt<scalar_t>(a_csr, a_colIndices, b_dense_tensor, c_dense_tensor, result_indices, result_vals);
  });

  //Create CPU sparse tensor (from SparseLLCopy):
  SparseTensor sparse_tensor = detail::make_tensor<SparseTensorImpl>(TensorTypeSet(TensorTypeId::SparseCPUTensorId), result_vals.options().dtype());
  get_sparse_impl(sparse_tensor)->resize_(a_sparse_tensor.sparse_dim(), a_sparse_tensor.dense_dim(), a_sparse_tensor.sizes());
  get_sparse_impl(sparse_tensor)->set_indices_and_values_unsafe(result_indices, result_vals);
  if(a_sparse_tensor.is_coalesced()) {
    get_sparse_impl(sparse_tensor)->set_coalesced(true);
  }
  return sparse_tensor;
}

// --------------------------------------------------------------------
// sddtmmd(SparseTensor, Tensor, Tensor) -> Tensor
//
// sddtmmd(a, b, c) returns (b@c.T) at locations were a is nonzero
// --------------------------------------------------------------------

template <typename scalar_t>
void sddtmmd_kernel_cpu(
  const SparseTensor& a_sparse_tensor,
  const Tensor& b_dense_tensor,
  const Tensor& c_dense_tensor,
  Tensor& out_tensor){

  if ( b_dense_tensor.scalar_type() != ScalarType::Float
    || c_dense_tensor.scalar_type() != ScalarType::Float ) {
    AT_ERROR("Sddtmmd is implemented for Float type only for matrices b and c"); 
  }

  TORCH_CHECK(a_sparse_tensor.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'a' in sddtmm!");
  TORCH_CHECK(b_dense_tensor.dim() == 2 && c_dense_tensor.dim() == 2, "Expected 2D matrixes for 'a' and 'b', but got ", b_dense_tensor.dim(), " and ", c_dense_tensor.dim(), " tensors");
  TORCH_CHECK(b_dense_tensor.size(1) == c_dense_tensor.size(1), "Matrix multiply dimension mismatch: 'b' dim 1 = ", b_dense_tensor.size(1), ", 'c'.T dim 0 = ", c_dense_tensor.size(1));

  TORCH_CHECK(b_dense_tensor.size(0) == a_sparse_tensor.size(0) && c_dense_tensor.size(0) == a_sparse_tensor.size(1),"SddTmm sample dimension mismatch: sample was shape ",a_sparse_tensor.size(0)," by ",a_sparse_tensor.size(1),", but b@c is shape ",b_dense_tensor.size(0)," by ",c_dense_tensor.size(0));

  auto indices = a_sparse_tensor._indices();
  TORCH_CHECK(indices.dtype() == at::kLong, "Indices should be long, but got ", indices.dtype());

  auto a_indices = indices.accessor<int64_t, 2>();
  auto b_dense = b_dense_tensor.accessor<scalar_t, 2>();
  auto c_dense = c_dense_tensor.accessor<scalar_t, 2>();
  auto out = out_tensor.accessor<scalar_t, 2>();

  int dot_len = b_dense.size(1);
  for (int k = 0; k < a_sparse_tensor._nnz(); k++) {
    int row = a_indices[0][k]; //0
    int col = a_indices[1][k];  //1

    float dot_total = 0;
    for (int i = 0; i < dot_len; i++)
      dot_total += b_dense[row][i] * c_dense[col][i];

    // update out
    out[row][col] = dot_total;
  }
}
SparseTensor sddtmmd_cpu(
  const SparseTensor& a_sparse_tensor,
  const Tensor& b_dense_tensor,
  const Tensor& c_dense_tensor) {
  
  Tensor result = at::zeros(a_sparse_tensor.sizes(), {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});

  AT_DISPATCH_ALL_TYPES(b_dense_tensor.scalar_type(), "sddtmmd_cpu", [&]{
    sddtmmd_kernel_cpu<scalar_t>(a_sparse_tensor, b_dense_tensor, c_dense_tensor, result);
  });

  return result;
}

// --------------------------------------------------------------------
// dstmm(Tensor, SparseTensor) -> Tensor
// 
// dstmm(a, b) returns a@b.T
// --------------------------------------------------------------------

template <typename scalar_t>
void dstmm_kernel(
    const Tensor& a_tensor, //dense
    const LongTensor& b_csc_tensor,
    const LongTensor& b_rows_tensor,
    const Tensor& b_vals_tensor,
    Tensor& res_tensor //destination
  ) {
  auto a = a_tensor.accessor<scalar_t, 2>();
  auto b_csc = b_csc_tensor.accessor<int64_t, 1>();
  auto b_rows = b_rows_tensor.accessor<int64_t, 1>();
  auto b_vals = b_vals_tensor.accessor<scalar_t, 1>();
  auto res = res_tensor.accessor<scalar_t, 2>();

  auto a_nrows = res.size(0);
  auto b_ncols = res.size(1);
  auto nnz = b_vals.size(0);

  float sum;
  for (int a_row = 0; a_row < a_nrows; a_row++){
    for (int b_col = 0; b_col < b_ncols; b_col++){
      sum = 0;
      for (int b_row_idx = b_csc[b_col]; b_row_idx < b_csc[b_col+1]; b_row_idx++){
        int b_row = b_rows[b_row_idx];
        float b_val = b_vals[b_row_idx];
        sum += b_val * a[a_row][b_row]; 
      }
      res[a_row][b_col] = sum;
    }
  }
}
Tensor dstmm_cpu(
  const Tensor& a_dense,
  const SparseTensor& bT_sparse) {

  if ( a_dense.scalar_type() != ScalarType::Float
    || bT_sparse.scalar_type() != ScalarType::Float ) {
    AT_ERROR("dstmm is implemented for Float only"); 
  }

  TORCH_CHECK(bT_sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'b' (sparse) in DenseSparseTMm!");
  TORCH_CHECK(a_dense.dim() == 2 && bT_sparse.dim() == 2, "Expected 2D matrixes for 'a' and 'b', but got ", a_dense.dim(), " and ", bT_sparse.dim(), " tensors");
  TORCH_CHECK(a_dense.size(1) == bT_sparse.size(1), "Matrix multiply dimension mismatch: 'a' dim 1 = ", a_dense.size(1), ", 'b' dim 0 = ", bT_sparse.size(1));

  LongTensor indices = bT_sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kLong, "Indices should be long, but got ", indices.dtype());

  LongTensor b_rowIndices = indices.select(0, 1);
  int64_t* b_colIndices = indices.select(0, 0).data_ptr<int64_t>();
  int64_t b_nnz = bT_sparse._nnz();
  int64_t b_dim = bT_sparse.size(0);
  LongTensor b_csc = _to_csr(b_colIndices, b_dim, b_nnz);
  Tensor b_values = bT_sparse._values();

  Tensor out_dense = at::zeros({a_dense.size(0), bT_sparse.size(0)}, {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});

  AT_DISPATCH_ALL_TYPES(a_dense.scalar_type(), "dstmm_cpu", [&]{
    dstmm_kernel<scalar_t>(a_dense, b_csc, b_rowIndices, b_values, out_dense);
  });

  return out_dense;
}

// --------------------------------------------------------------------
// dstmmt(Tensor, SparseTensor) -> Tensor
//
// dstmmt(a, b) returns (a@b.T).T
// --------------------------------------------------------------------

template <typename scalar_t>
void dstmmt_kernel(
    const Tensor& a_tensor, //dense
    const LongTensor& b_csc_tensor,
    const LongTensor& b_rows_tensor,
    const Tensor& b_vals_tensor,
    Tensor& res_tensor //destination
  ) {
  auto a = a_tensor.accessor<scalar_t, 2>();
  auto b_csc = b_csc_tensor.accessor<int64_t, 1>();
  auto b_rows = b_rows_tensor.accessor<int64_t, 1>();
  auto b_vals = b_vals_tensor.accessor<scalar_t, 1>();
  auto res = res_tensor.accessor<scalar_t, 2>();

  auto a_nrows = res.size(1);
  auto b_ncols = res.size(0);
  auto nnz = b_vals.size(0);

  float sum;
  for (int a_row = 0; a_row < a_nrows; a_row++){
    for (int b_col = 0; b_col < b_ncols; b_col++){
      sum = 0;
      for (int b_row_idx = b_csc[b_col]; b_row_idx < b_csc[b_col+1]; b_row_idx++){
        int b_row = b_rows[b_row_idx];
        float b_val = b_vals[b_row_idx];
        sum += b_val * a[a_row][b_row]; 
      }
      res[b_col][a_row] = sum;
    }
  }
}
Tensor dstmmt_cpu(
  const Tensor& a_dense,
  const SparseTensor& bT_sparse) {

  if ( a_dense.scalar_type() != ScalarType::Float
    || bT_sparse.scalar_type() != ScalarType::Float ) {
    AT_ERROR("dstmmt is implemented for Float only"); 
  }

  TORCH_CHECK(bT_sparse.sparse_dim() == 2, "We do not support hybrid sparse tensor for 'b' (sparse) in DenseSparseTMm!");
  TORCH_CHECK(a_dense.dim() == 2 && bT_sparse.dim() == 2, "Expected 2D matrixes for 'a' and 'b', but got ", a_dense.dim(), " and ", bT_sparse.dim(), " tensors");
  TORCH_CHECK(a_dense.size(1) == bT_sparse.size(1), "Matrix multiply dimension mismatch: 'a' dim 1 = ", a_dense.size(1), ", 'b' dim 0 = ", bT_sparse.size(1));

  LongTensor indices = bT_sparse._indices();
  TORCH_CHECK(indices.dtype() == at::kLong, "Indices should be long, but got ", indices.dtype());

  LongTensor b_rowIndices = indices.select(0, 1);
  int64_t* b_colIndices = indices.select(0, 0).data_ptr<int64_t>();
  int64_t b_nnz = bT_sparse._nnz();
  int64_t b_dim = bT_sparse.size(0);
  LongTensor b_csc = _to_csr(b_colIndices, b_dim, b_nnz);
  Tensor b_values = bT_sparse._values();

  Tensor out_dense = at::zeros({bT_sparse.size(0), a_dense.size(0)}, {at::requires_grad().device(at::kCPU).dtype(at::kFloat)});

  AT_DISPATCH_ALL_TYPES(a_dense.scalar_type(), "dstmmt_cpu", [&]{
    dstmmt_kernel<scalar_t>(a_dense, b_csc, b_rowIndices, b_values, out_dense);
  });

  return out_dense;
}


// --------------------------------------------------------------------
// sreciprocal_(SparseTensor) -> SparseTensor
//
// sreciprocal_(a) returns 1.0 / a, ignoring a's zero entries
// ** and modifies a **
// a must be a coalesced tensor
// --------------------------------------------------------------------

SparseTensor& _sampled_reciprocal__cpu(SparseTensor& self) {
  TORCH_CHECK(self.is_coalesced(), "_sampled_reciprocal__cpu only supports a coalesced tensor");
  int64_t nnz = self._nnz(); // = vals.numel()
  Tensor vals = self._values();

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "_sampled_reciprocal__cpu", [&]{
    auto out = vals.accessor<scalar_t, 1>();
    for (int i = 0; i < nnz; i++){
      out[i] = 1 / out[i];
    }
  });

  return self;
}


}} // namespace at::native
