//====================================================================
// Helper functions for SPMD 2D Conv
// 10/06/2020 Lin Cheng
//====================================================================

// templated unrolling code by Krithik Ranjan
template<int N, typename T>
struct Unroll {
  inline static void reset_buffer(T* buf);
  inline static void fill_buffer(T* src, T* buf);
  inline static void drain_buffer(T* buf, T* dest);
};

template<int N, typename T>
inline void Unroll<N, T>::reset_buffer(T* buf) {
  buf[N] = 0;
  Unroll<N-1, T>::reset_buffer(buf);
}

template<int N, typename T>
inline void Unroll<N, T>::fill_buffer(T* src, T* buf) {
  buf[N] = ((bsg_attr_remote T*) src)[N];
  Unroll<N-1, T>::fill_buffer(src, buf);
}

template<int N, typename T>
inline void Unroll<N, T>::drain_buffer(T* buf, T* dest) {
  ((bsg_attr_remote T*) dest)[N] = buf[N];
  Unroll<N-1, T>::drain_buffer(buf, dest);
}

template<typename T>
struct Unroll<0, T> {
  inline static void reset_buffer(T* buf);
  inline static void fill_buffer(T* src, T* buf);
  inline static void drain_buffer(T* buf, T* dest);
};

template<typename T>
inline void Unroll<0, T>::reset_buffer(T* buf) {
  buf[0] = 0;
}

template<typename T>
inline void Unroll<0, T>::fill_buffer(T* src, T* buf) {
  buf[0] = ((bsg_attr_remote T*) src)[0];
}

template<typename T>
inline void Unroll<0, T>::drain_buffer(T* buf, T* dest) {
  ((bsg_attr_remote T*) dest)[0] = buf[0];
}

// conv related helpers

template<int DIM_X, int DIM_Y>
inline void reset_buffer(float* buf) {
  for (size_t i = 0; i < DIM_Y; i++) {
    Unroll<DIM_X-1, float>::reset_buffer(buf);
    buf += DIM_X;
  }
}

template<int DIM>
inline void fill_filter_buffer(float* src, float* buf) {
  Unroll<DIM*DIM-1, float>::fill_buffer(src, buf);
}

template<int DIM_X, int DIM_Y>
inline void fill_imap_buffer(float* src, float* buf, size_t y_step) {
  for (size_t i = 0; i < DIM_Y; i++) {
    Unroll<DIM_X-1, float>::fill_buffer(src, buf);
    buf += DIM_X;
    src += y_step;
  }
}

template<int DIM_X, int DIM_Y>
inline void drain_omap_buffer(float* buf, float* dest, size_t y_step) {
  for (size_t i = 0; i < DIM_Y; i++) {
    Unroll<DIM_X-1, float>::drain_buffer(buf, dest);
    buf += DIM_X;
    dest += y_step;
  }
}


inline void conv2d_5x5(float* imap, float* filter, float* omap) {
  for (size_t y = 0; y < BLOCK_DIM_Y; y++) {
    for (size_t x = 0; x < BLOCK_DIM_X; x += 7) {
      register float psum0 = omap[y * BLOCK_DIM_X + x + 0];
      register float psum1 = omap[y * BLOCK_DIM_X + x + 1];
      register float psum2 = omap[y * BLOCK_DIM_X + x + 2];
      register float psum3 = omap[y * BLOCK_DIM_X + x + 3];
      register float psum4 = omap[y * BLOCK_DIM_X + x + 4];
      register float psum5 = omap[y * BLOCK_DIM_X + x + 5];
      register float psum6 = omap[y * BLOCK_DIM_X + x + 6];
      for (size_t yy = 0; yy < FILTER_DIM; yy++) {
        register float filter0 = filter[yy * FILTER_DIM + 0];
        register float filter1 = filter[yy * FILTER_DIM + 1];
        register float filter2 = filter[yy * FILTER_DIM + 2];
        register float filter3 = filter[yy * FILTER_DIM + 3];
        register float filter4 = filter[yy * FILTER_DIM + 4];
        register float imap0  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 0];
        register float imap1  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 1];
        register float imap2  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 2];
        register float imap3  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 3];
        register float imap4  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 4];
        register float imap5  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 5];
        register float imap6  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 6];
        register float imap7  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 7];
        register float imap8  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 8];
        register float imap9  = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 9];
        register float imap10 = imap[y * IMAP_DIM_X + x + yy * IMAP_DIM_X + 10];
        asm volatile("": : :"memory");

#ifdef HB_EMUL
        psum0 += imap0 * filter0;
        psum1 += imap1 * filter0;
        psum2 += imap2 * filter0;
        psum3 += imap3 * filter0;
        psum4 += imap4 * filter0;
        psum5 += imap5 * filter0;
        psum6 += imap6 * filter0;

        psum0 += imap1 * filter1;
        psum1 += imap2 * filter1;
        psum2 += imap3 * filter1;
        psum3 += imap4 * filter1;
        psum4 += imap5 * filter1;
        psum5 += imap6 * filter1;
        psum6 += imap7 * filter1;

        psum0 += imap2 * filter2;
        psum1 += imap3 * filter2;
        psum2 += imap4 * filter2;
        psum3 += imap5 * filter2;
        psum4 += imap6 * filter2;
        psum5 += imap7 * filter2;
        psum6 += imap8 * filter2;

        psum0 += imap3 * filter3;
        psum1 += imap4 * filter3;
        psum2 += imap5 * filter3;
        psum3 += imap6 * filter3;
        psum4 += imap7 * filter3;
        psum5 += imap8 * filter3;
        psum6 += imap9 * filter3;

        psum0 += imap4 * filter4;
        psum1 += imap5 * filter4;
        psum2 += imap6 * filter4;
        psum3 += imap7 * filter4;
        psum4 += imap8 * filter4;
        psum5 += imap9 * filter4;
        psum6 += imap10 * filter4;
#else
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0) : "f"(imap0), "f"(filter0));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1) : "f"(imap1), "f"(filter0));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2) : "f"(imap2), "f"(filter0));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3) : "f"(imap3), "f"(filter0));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4) : "f"(imap4), "f"(filter0));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5) : "f"(imap5), "f"(filter0));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6) : "f"(imap6), "f"(filter0));

        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0) : "f"(imap1), "f"(filter1));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1) : "f"(imap2), "f"(filter1));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2) : "f"(imap3), "f"(filter1));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3) : "f"(imap4), "f"(filter1));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4) : "f"(imap5), "f"(filter1));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5) : "f"(imap6), "f"(filter1));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6) : "f"(imap7), "f"(filter1));

        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0) : "f"(imap2), "f"(filter2));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1) : "f"(imap3), "f"(filter2));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2) : "f"(imap4), "f"(filter2));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3) : "f"(imap5), "f"(filter2));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4) : "f"(imap6), "f"(filter2));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5) : "f"(imap7), "f"(filter2));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6) : "f"(imap8), "f"(filter2));

        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0) : "f"(imap3), "f"(filter3));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1) : "f"(imap4), "f"(filter3));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2) : "f"(imap5), "f"(filter3));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3) : "f"(imap6), "f"(filter3));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4) : "f"(imap7), "f"(filter3));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5) : "f"(imap8), "f"(filter3));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6) : "f"(imap9), "f"(filter3));

        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum0) : "f"(imap4), "f"(filter4));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum1) : "f"(imap5), "f"(filter4));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum2) : "f"(imap6), "f"(filter4));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum3) : "f"(imap7), "f"(filter4));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum4) : "f"(imap8), "f"(filter4));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum5) : "f"(imap9), "f"(filter4));
        asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum6) : "f"(imap10), "f"(filter4));
#endif
      }
      omap[y * BLOCK_DIM_X + x + 0] = psum0;
      omap[y * BLOCK_DIM_X + x + 1] = psum1;
      omap[y * BLOCK_DIM_X + x + 2] = psum2;
      omap[y * BLOCK_DIM_X + x + 3] = psum3;
      omap[y * BLOCK_DIM_X + x + 4] = psum4;
      omap[y * BLOCK_DIM_X + x + 5] = psum5;
      omap[y * BLOCK_DIM_X + x + 6] = psum6;
    }
  }
}
