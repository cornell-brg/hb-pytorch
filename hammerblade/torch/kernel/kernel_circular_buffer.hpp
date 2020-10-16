//----------------------------------------------------
// A circular buffer implementation -- based on the work by Dustin
// The main points of this are two folds:
//   (1) make systolic style programming simpler
//   (2) make emulation of remote scratchpad possible
// Compare to Dustin's work, this circular buffer does not rely on
// busy wait
//
// 03/05/2020 Dustin Richmond
// 10/14/2020 Lin Cheng
//----------------------------------------------------

#ifndef __HB_CIRCULAR_BUFFER_HPP
#define __HB_CIRCULAR_BUFFER_HPP

namespace CircularBuffer{

  template<typename T, unsigned int N, unsigned int DEPTH = 4>
  class FIFO{
    protected:
      unsigned int occ_idx = 0;
      unsigned int occ_idx_nxt = 0;
      volatile unsigned int      occupancy [DEPTH] = {0};
      volatile unsigned int  occupancy_nxt [DEPTH] = {0};
      volatile unsigned int* occupancy_nxt_r = nullptr;
      volatile unsigned int* occupancy_prv_r = nullptr;
      T buffer[N * DEPTH];
      T* buffer_remote = nullptr;
    public:
      __attribute__((always_inline))
      FIFO<T, N, DEPTH>(unsigned int prv_y, unsigned int prv_x, unsigned int nxt_y, unsigned int nxt_x){
        // I know this is super confusing ...
        // But trust me -- this is what it should look like
        occupancy_prv_r = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(prv_x,prv_y,occupancy_nxt));
        occupancy_nxt_r = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(nxt_x,nxt_y,occupancy));
        buffer_remote = reinterpret_cast<T*>(bsg_tile_group_remote_pointer(nxt_x,nxt_y,buffer));
      }

      __attribute__((always_inline))
      T *obtain_wr_ptr(){
        // bsg_print_hexadecimal(0xFACEB00C);
        volatile unsigned int *o = &(occupancy_nxt[occ_idx_nxt]);
        bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (o)), 0);
        // bsg_print_hexadecimal(0xFACEB00D);
        return buffer_remote + occ_idx_nxt * N;
      }

      __attribute__((always_inline))
      int finish_wr_ptr(){
        volatile unsigned int *o   = &(occupancy_nxt[occ_idx_nxt]);
        volatile unsigned int *o_r = occupancy_nxt_r + occ_idx_nxt;
        asm volatile("": : :"memory");
        *o   = 1;
        *o_r = 1;
        occ_idx_nxt = (occ_idx_nxt + 1) % DEPTH;
        // bsg_print_hexadecimal(0xBEEFBEEF);
        return 0;
      }

      __attribute__((always_inline))
      T *obtain_rd_ptr(){
        // bsg_print_hexadecimal(0xFACE);
        volatile unsigned int *o = &(occupancy[occ_idx]);
        bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (o)), 1);
        // bsg_print_hexadecimal(0xFACF);
        return buffer + occ_idx * N;
      }

      __attribute__((always_inline))
      int finish_rd_ptr(){
        volatile unsigned int *o   = &(occupancy[occ_idx]);
        volatile unsigned int *o_r = occupancy_prv_r + occ_idx;
        asm volatile("": : :"memory");
        *o   = 0;
        *o_r = 0;
        occ_idx = (occ_idx + 1) % DEPTH;
        // bsg_print_hexadecimal(0xBEEF);
        return 0;
      }

      __attribute__((always_inline))
      T* get_buffer(){
        return buffer;
      }
  };

}

#endif
