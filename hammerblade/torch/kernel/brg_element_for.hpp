//====================================================================
// Element-wise for helper function
// 03/12/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================

#ifndef _BRG_ELEMENT_FOR_HPP
#define _BRG_ELEMENT_FOR_HPP

#include <map>
#include <math.h>
#include <initializer_list>
#include <bsg_assert.hpp>
#include <bsg_tensor.hpp>

// =========================================================
// Device Tensor that comes from TensorIterator, in which its
// strides are measured in **bytes**
//
// Helper class to make our lives easier when writing element
// -wise operation kernels
// Note: here strides could be *0*
//
// You may call ++ to advance the iterator, but be noted that
// it is monolithic. There is no way to return
// =========================================================

template <typename T>
class BRGIteratorTensor {
  private:
    uint32_t N;
    uint32_t dims;
    uint32_t strides;
#ifdef HB_EMUL
    uint64_t data;
#else
    uint32_t data;
#endif
    uint32_t cur_loc;

  public:
    BRGIteratorTensor(bsg_tensor_t* t, uint32_t start_loc = 0) :
      N(t->N),
      dims(t->dims),
      strides(*(uint32_t*)((intptr_t) t->strides)),
      data((intptr_t) t->data),
      cur_loc(start_loc) {
        bsg_assert(cur_loc < N);
        data += start_loc * strides;
        bsg_assert(dims == 1);
      }

    uint32_t numel() {
      return N;
    }

    //-----------------
    // post increment
    //-----------------
    BRGIteratorTensor& operator ++ (int) {
      data += strides;
      cur_loc++;
      return *this;
    }

    //-----------------------------------
    // *accessor, just like c++ iterators
    //-----------------------------------
    T operator*() {
      bsg_assert(cur_loc < N);
      return (T)(intptr_t)data;
    }

};

// =========================================================
// Helper function to get the argument type of lambda function
//
// Creazy meta-programming in here ...
// Please refer to https://stackoverflow.com/questions/28105371/
// is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-polymorphic
// and
// http://coliru.stacked-crooked.com/a/e19151a5c245d7c3
//==========================================================

namespace function_traits {
  //--------------------------
  // for lambda and functors
  //--------------------------
  template <typename T>
  struct traits : public traits<decltype(&T::operator())>{};

  template <typename ClassType, typename ReturnType, typename... Args>
  struct traits<ReturnType(ClassType::*)(Args...) const> {
    using result_type = ReturnType;
    template <size_t i> struct arg {
        using type = typename std::tuple_element<i, std::tuple<Args...>>::type;
    };
  };

  //--------------------------
  // for function pointers
  //--------------------------
  template <typename ReturnType, typename... Args>
  struct traits<ReturnType(*)(Args...)> : public traits<ReturnType(Args...)>{};

  template <typename ReturnType, typename... Args>
  struct traits<ReturnType(Args...)> {
    using result_type = ReturnType;
    template <size_t i> struct arg {
        using type = typename std::tuple_element<i, std::tuple<Args...>>::type;
    };
  };
}

// =========================================================
// Element-wise for -- Binary ops
//
// This function iterates over all elements, starting from element 0
//
// We would like the synatx to be like this:
// __attribute__ ((noinline))  int tensorlib_add(bsg_tensor_t* c_p,
//                                               bsg_tensor_t* a_p,
//                                               bsg_tensor_t* b_p,
//                                               float* alpha) {
//    brg_element_wise_for(t0_p, t1_p, t2_p,
//        [&](float a, float b) {
//          return a + alpha * b;
//        });
// }
//==========================================================

template <class FetchFunctor>
inline void brg_element_wise_for(bsg_tensor_t* _t0, bsg_tensor_t* _t1,
                                 bsg_tensor_t* _t2, FetchFunctor functor) {
  //--------------------------------------------------
  // get the type of frist argument of lambda function
  //-------------------------------------------------
  using f = function_traits::traits<decltype(functor)>;
  using T = typename f::template arg<0>::type;
  //-----------------
  // wrap bsg_tensors
  //-----------------
  auto res = BRGIteratorTensor<T*>(_t0);
  auto input = BRGIteratorTensor<T*>(_t1);
  auto other = BRGIteratorTensor<T*>(_t2);
  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t start = 0;
  size_t end = res.numel();
  for (size_t i = start; i < end; i++) {
    *(*res) = functor(*(*input), *(*other));
    res++;
    input++;
    other++;
  }
}

// =========================================================
// Element-wise for -- Unary ops
//==========================================================

template <class FetchFunctor>
inline void brg_element_wise_for(bsg_tensor_t* _t0, bsg_tensor_t* _t1,
                                 FetchFunctor functor) {
  //--------------------------------------------------
  // get the type of frist argument of lambda function
  //-------------------------------------------------
  using f = function_traits::traits<decltype(functor)>;
  using T = typename f::template arg<0>::type;
  //-----------------
  // wrap bsg_tensors
  //-----------------
  auto res = BRGIteratorTensor<T*>(_t0);
  auto input = BRGIteratorTensor<T*>(_t1);
  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t start = 0;
  size_t end = res.numel();
  for (size_t i = start; i < end; i++) {
    *(*res) = functor(*(*input));
    res++;
    input++;
  }
}

// =========================================================
// Element-wise for -- Nullary ops
//==========================================================

template <class FetchFunctor>
inline void brg_element_wise_for(bsg_tensor_t* _t0,
                                 FetchFunctor functor) {
  //--------------------------------------------------
  // get the type of frist argument of lambda function
  //-------------------------------------------------
  using f = function_traits::traits<decltype(functor)>;
  using T = typename f::template arg<0>::type;
  //-----------------
  // wrap bsg_tensors
  //-----------------
  auto res = BRGIteratorTensor<T*>(_t0);
  //-----------------------------
  // iterating over all elementes
  //-----------------------------
  size_t start = 0;
  size_t end = res.numel();
  for (size_t i = start; i < end; i++) {
    *(*res) = functor();
    res++;
  }
}



// =========================================================
// Tile Element-wise for -- Binary ops
//
// This function calculates the per tile range automatically
//==========================================================

template <class FetchFunctor>
inline void brg_tile_element_wise_for(bsg_tensor_t* _t0, bsg_tensor_t* _t1,
                                      bsg_tensor_t* _t2, FetchFunctor functor) {
  //--------------------------------------------------
  // get the type of frist argument of lambda function
  //-------------------------------------------------
  using f = function_traits::traits<decltype(functor)>;
  using T = typename f::template arg<0>::type;
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  size_t len_per_tile = _t0->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > _t0->N)  ? _t0->N : end;
  //-----------------
  // wrap bsg_tensors
  //-----------------
  auto res = BRGIteratorTensor<T*>(_t0, start);
  auto input = BRGIteratorTensor<T*>(_t1, start);
  auto other = BRGIteratorTensor<T*>(_t2, start);
  for (size_t i = start; i < end; i++) {
    *(*res) = functor(*(*input), *(*other));
    res++;
    input++;
    other++;
  }
}


// =========================================================
// Tile Element-wise for -- Unary ops
//==========================================================

template <class FetchFunctor>
inline void brg_tile_element_wise_for(bsg_tensor_t* _t0, bsg_tensor_t* _t1,
                                      FetchFunctor functor) {
  //--------------------------------------------------
  // get the type of frist argument of lambda function
  //-------------------------------------------------
  using f = function_traits::traits<decltype(functor)>;
  using T = typename f::template arg<0>::type;
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  size_t len_per_tile = _t0->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > _t0->N)  ? _t0->N : end;
  //-----------------
  // wrap bsg_tensors
  //-----------------
  auto res = BRGIteratorTensor<T*>(_t0, start);
  auto input = BRGIteratorTensor<T*>(_t1, start);
  for (size_t i = start; i < end; i++) {
    *(*res) = functor(*(*input));
    res++;
    input++;
  }
}


// =========================================================
// Tile Element-wise for -- Nullary ops
//==========================================================

template <class FetchFunctor>
inline void brg_tile_element_wise_for(bsg_tensor_t* _t0,
                                      FetchFunctor functor) {
  //--------------------------------------------------
  // get the type of frist argument of lambda function
  //-------------------------------------------------
  using f = function_traits::traits<decltype(functor)>;
  using T = typename f::template arg<0>::type;
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  size_t len_per_tile = _t0->N / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > _t0->N)  ? _t0->N : end;
  //-----------------
  // wrap bsg_tensors
  //-----------------
  auto res = BRGIteratorTensor<T*>(_t0, start);
  for (size_t i = start; i < end; i++) {
    *(*res) = functor();
    res++;
  }
}

// =========================================================
// BRG for
// =========================================================
// functor takes in current index

template <class FetchFunctor>
inline void brg_for(size_t numel, FetchFunctor functor) {
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  size_t start = 0;
  size_t end = numel;
  //-----------------
  // loop
  //----------------
  for (size_t i = start; i < end; i++) {
    functor(i);
  }
}

// =========================================================
// BRG tile for
// =========================================================
// functor takes in current index

template <class FetchFunctor>
inline void brg_tile_for(size_t numel, FetchFunctor functor) {
  //--------------------------------------
  // calculate start and end for this tile
  //--------------------------------------
  size_t len_per_tile = numel / (bsg_tiles_X * bsg_tiles_Y) + 1;
  size_t start = len_per_tile * __bsg_id;
  size_t end = start + len_per_tile;
  end = (end > numel)  ? numel : end;
  //-----------------
  // loop
  //----------------
  for (size_t i = start; i < end; i++) {
    functor(i);
  }
}


#endif
