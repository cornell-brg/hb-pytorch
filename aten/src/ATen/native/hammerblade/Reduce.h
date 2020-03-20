#pragma once

#include <ATen/Parallel.h>
#include <c10/util/C++17.h>
#include <c10/util/TypeList.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native { namespace {


template<typename traits, typename res_t>
static void set_result(const int index, const res_t result, const TensorIterator &iter, const int num_outputs) {
  // static_assert(std::is_same<res_t, typename traits::arg2_t>::value, "data types must match");
  if (index < num_outputs) {
    char *out = (char *) iter.data_ptr(index);
    *(res_t *) out = result;
  }
}

template<typename traits, typename res_t>
static void set_results(const res_t result, const TensorIterator &iter, const int num_outputs) {
  AT_ASSERT(num_outputs == 1);
  set_result<traits>(0, result, iter, num_outputs);
}

template <typename T, typename... Args>
struct all_same : c10::guts::conjunction<
  std::is_same<T, Args>...
> {};

// data_t is the input/output data type.
// acc_t is a type that contains all the necessary data
// to continue reducing.
// index_t is a one-dimensional index
//
// ops_t is such that &ops_t::reduce, &ops_t::combine, and &ops_t::project exist and satisfy
// the following.
// reduce: (acc_t, data_t, index_t) -> acc_t adds one data point to the accumulated value.
// combine: (acc_t, acc_t) -> acc_t combines two accumulated values into one.
// project: acc_t -> out_t finishes the reduction, getting the required output.
//
// Additionally, acc_t must be default-constructible:
// acc_t {} is an identity for combine,
// and project(acc_t {}) is the value of the operation on zero elements.
//
// The point of `combine` is to support parallelization -
// the idea is to one sequence of `reduce` calls per thread of execution,
// and then to combine them at the end with `combine`.
//
// If there is more than one output element,
// our parallelization strategy is to use one thread for each of them,
// which means that `combine` will never be called.
//
// If, on the other hand, there is only one, then we split the input into
// into several pieces, reduce each separately, and then combine them.

template <typename ops_t, typename init_t>
void binary_kernel_reduce(TensorIterator& iter, ops_t ops, init_t init) {
  using rf_t = decltype(&ops_t::reduce);
  using cf_t = decltype(&ops_t::combine);
  using pf_t = decltype(&ops_t::project);
  using r_traits = binary_function_traits<rf_t>;
  using c_traits = binary_function_traits<cf_t>;
  using p_traits = unary_function_traits<pf_t>;
  using acc_t = typename p_traits::arg1_t;
  using data_t = typename r_traits::arg2_t;
  static_assert(
    all_same<
      acc_t,
      init_t,
      typename r_traits::arg1_t,
      typename r_traits::result_type,
      typename c_traits::arg1_t,
      typename c_traits::arg2_t,
      typename c_traits::result_type>::value,
    "all accumulate types must match");
  static_assert(
    std::is_default_constructible<acc_t>::value,
    "the accumulate type must be default-constructible"
  );
  const int num_outputs = iter.noutputs();
  iter.foreach_reduced_elt([&ops, &init, num_outputs](TensorIterator &sub_iter) {
    std::cout << "functor inside iter.foreach_reduced_elt called with init = " << std::endl;
    auto reduction_body = [&ops, &sub_iter, num_outputs](acc_t acc, int64_t begin, int64_t end) -> acc_t {
      int ntensors = sub_iter.ntensors();
      sub_iter.serial_for_each([&acc, &ops, num_outputs, ntensors, begin](char** data, const int64_t* strides, int64_t size) {
        AT_ASSERT(ntensors - num_outputs == 1);
        char *in = data[ntensors - 1];
        int64_t stride = strides[ntensors - 1];
        for (int64_t i = 0; i < size; ++i) {
          acc = ops.reduce(acc, *(data_t*)in, begin + i);
          in += stride;
        }
      }, {begin, end});
      return acc;
    };
    acc_t total_acc = init;
    auto numel = sub_iter.numel();
    if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
        at::in_parallel_region()) {
      total_acc = reduction_body(total_acc, 0, numel);
    } else {
      int max_threads = at::get_num_threads();
      AT_ASSERT(max_threads > 0);
      static_assert(
        !std::is_same<acc_t, bool>::value,
        "Concurrently modifying different references into std::vector<bool> is UB."
      );
      std::vector<acc_t> buffer((unsigned)max_threads, init);
      at::parallel_for(0, numel, internal::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
          auto& acc = buffer[at::get_thread_num()];
          acc = reduction_body(acc, begin, end);
        }
      );
      for (int i = 0; i < max_threads; ++i) {
        total_acc = ops.combine(total_acc, buffer[i]);
      }
    }
    set_results<r_traits>(ops.project(total_acc), sub_iter, num_outputs);
  });
}

}}}  // namespace at::native::<anonymous>
