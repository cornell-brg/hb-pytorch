//====================================================================
// Element-wise mul and div kernel
// 03/05/2020 Lin Cheng and Bandhav Veluri (lc873@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

template<int N, typename scalar_t, typename F>
struct Unroll {
  inline static void copy_from(scalar_t* src, scalar_t* dest, size_t i);
  inline static void copy_to(scalar_t* src, scalar_t* dest, size_t i);
  inline static void compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor);
};

template<int N, typename scalar_t, typename F>
inline void Unroll<N, scalar_t, F>::copy_from(scalar_t* src, scalar_t* dest, size_t i){
  dest[N] = src[i + N];
  Unroll<N-1, scalar_t, F>::copy_from(src, dest, i);
}

template<int N, typename scalar_t, typename F>
inline void Unroll<N, scalar_t, F>::copy_to(scalar_t* src, scalar_t* dest, size_t i){
  dest[N + i] = src[N];
  Unroll<N-1, scalar_t, F>::copy_to(src, dest, i);
}

template<int N, typename scalar_t, typename F>
inline void Unroll<N, scalar_t, F>::compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor){
  res[N] = functor(x[N], y[N]);
  Unroll<N-1, scalar_t, F>::compute(res, x, y, functor);
}

template<typename scalar_t, typename F>
struct Unroll<0, scalar_t, F> {
  inline static void copy_from(scalar_t* src, scalar_t* dest, size_t i);
  inline static void copy_to(scalar_t* src, scalar_t* dest, size_t i);
  inline static void compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor);
};

template<typename scalar_t, typename F>
inline void Unroll<0, scalar_t, F>::copy_from(scalar_t* src, scalar_t* dest, size_t i){
  dest[0] = src[i];
}

template<typename scalar_t, typename F>
inline void Unroll<0, scalar_t, F>::copy_to(scalar_t* src, scalar_t* dest, size_t i){
  dest[i] = src[0];
}

template<typename scalar_t, typename F>
inline void Unroll<0, scalar_t, F>::compute(scalar_t* res, scalar_t* x, scalar_t* y, F functor){
  res[0] = functor(x[0], y[0]);
}


template< int N, typename scalar_t, typename F>
inline void hb_tiled_foreach_unroll_pragma(HBTensor<scalar_t> result,
                               HBTensor<scalar_t> input,
                               HBTensor<scalar_t> other,
                               F functor) {
  scalar_t* data[3];
  data[0] = (scalar_t*)result.data_ptr();
  data[1] = (scalar_t*)input.data_ptr();
  data[2] = (scalar_t*)other.data_ptr();

  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t len_per_tile = result.numel() / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > result.numel())  ? result.numel() : end;

  // is_trivial_1d
  if(result.ndim() == 1) {

    //-----------------------------
    // collect metadata
    //-----------------------------
    uint32_t strides[3];
    strides[0] = (result.get_strides())[0];
    strides[1] = (input.get_strides())[0];
    strides[2] = (other.get_strides())[0];

    register scalar_t x[N];
    register scalar_t y[N];
    register scalar_t res[N];

    size_t i = start;
    while (i + N < end) {
      
      Unroll<N-1, scalar_t, F>::copy_from(data[1], y, i);
      Unroll<N-1, scalar_t, F>::copy_from(data[2], x, i);
      
      
      /*
      Unroll<N-1, scalar_t, F>::copy_from(&input, y, i);
      Unroll<N-1, scalar_t, F>::copy_from(&other, x, i);
      */
      Unroll<N-1, scalar_t, F>::compute(res, x, y, functor);
      
      Unroll<N-1, scalar_t, F>::copy_to(res, data[0], i);
      //Unroll<N-1, scalar_t, F>::copy_to(res, &result, i);

      i += N;
    }
    if (start + N < end) {
      i -= N;
    }
    while (i < end) {
      
      scalar_t x = *(data[1] + i * strides[1]);
      scalar_t y = *(data[2] + i * strides[2]);
      scalar_t res = functor(x, y);
      *(data[0] + strides[0]*i++) = res;
    }
  }
  /*
  else {
    size_t i = start;
    while (i < end) {
      scalar_t x = *(data[1] + offset(i, input));
      scalar_t y = *(data[2] + offset(i, other));
      scalar_t res = functor(x, y);
      *(data[0] + offset(i, result)) = res;
      i++;
    }
  } 
  */
}


extern "C" {

  __attribute__ ((noinline))  int tensorlib_mul(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);

    bsg_cuda_print_stat_kernel_start();

    /*
    hb_tiled_foreach(c, a, b,
        [&](float a, float b) {
          return a * b;
        });
    */

    hb_tiled_foreach_unroll_pragma<6>(c, a, b,
        [&](float a, float b) {
          return a * b;
        });
    

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mul, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)


  __attribute__ ((noinline))  int tensorlib_div(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p) {
    auto c = HBTensor<float>(t0_p);
    auto a = HBTensor<float>(t1_p);
    auto b = HBTensor<float>(t2_p);

    bsg_cuda_print_stat_kernel_start();

    hb_tiled_foreach(c, a, b,
      [&](float a, float b) {
        return a / b;
    });

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_div, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
