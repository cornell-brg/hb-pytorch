//====================================================================
// Convolution kernel
// 27/07/2020 Bandhav Veluri
//====================================================================

#ifndef _KERNEL_CONV_HPP
#define _KERNEL_CONV_HPP

template<typename F>
void blocked_for(size_t N, F functor) {
  size_t group_size, start, end;

  if(N >= bsg_tiles_X * bsg_tiles_Y) {
    size_t split = N / (bsg_tiles_X * bsg_tiles_Y) + 1;
    group_size = 1;
    start = split * __bsg_id;
    end = start + split;
    end = (end > N) ? N : end;
  } else {
    group_size = (bsg_tiles_X * bsg_tiles_Y) / N;
    start = __bsg_id / group_size;
    end = (start >= N) ? start : start + 1;
  }

  for(size_t i = start; i < end; ++i)
    functor(i, group_size);
}

template <class FetchFunctor>
inline void blocked_tiled_for(FetchFunctor functor,
                              size_t group_size, size_t id,
                              size_t O, size_t N, size_t M) {
  size_t numel = O * N * M;

  // per tile range within a pod
  size_t len_per_tile = numel / group_size + 1;
  size_t start        = len_per_tile * id;
  size_t end          = start + len_per_tile;
  end = (end > numel) ? numel : end;

  //-----------------
  // loop
  //----------------
  for (size_t i = start; i < end; i++) {
    size_t c = (i / (N * M)) % O;
    size_t b = (i / M) % N;
    size_t a = i % M;
    functor(c, b, a);
  }
}

#endif // _KERNEL_CONV_HPP
