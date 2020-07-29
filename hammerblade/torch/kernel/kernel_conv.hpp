//====================================================================
// Convolution kernel
// 27/07/2020 Bandhav Veluri
//====================================================================

#ifndef _KERNEL_CONV_HPP
#define _KERNEL_CONV_HPP

template<typename F>
void blocked_for(size_t tg_size, size_t N, F functor) {
  size_t group_size, start, end;
  size_t tile_id = __bsg_id % tg_size;

  if(N >= tg_size) {
    size_t split = N / tg_size + 1;
    group_size = 1;
    start = split * tile_id;
    end = start + split;
    end = (end > N) ? N : end;
  } else {
    group_size = tg_size / N;
    start = tile_id / group_size;
    end = (start >= N) ? start : start + 1;
  }

  for(size_t i = start; i < end; ++i)
    functor(i, group_size);
}

#endif // _KERNEL_CONV_HPP
