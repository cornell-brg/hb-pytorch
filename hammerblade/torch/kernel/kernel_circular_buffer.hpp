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
      volatile unsigned int      occupancy [DEPTH] = {{0}};
      volatile unsigned int  occupancy_nxt [DEPTH] = {{0}};
      volatile unsigned int* occupancy_nxt_r = nullptr;
      volatile unsigned int* occupancy_prv_r = nullptr;
      T buffer[N * DEPTH];
      T* buffer_remote = nullptr;
    public:
      __attribute__((always_inline))
      FIFO<T, N, DEPTH>(unsigned int prv_y, unsigned int prv_x, unsigned int nxt_y, unsigned int nxt_x){
        // I know this is super confusing ...
        // But trust me -- this is what is should look like
        this->occupancy_prv_r = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(prv_x,prv_y,this->occupancy_nxt));
        this->occupancy_nxt_r = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(nxt_x,nxt_y,this->occupancy));
        this->buffer_remote = reinterpret_cast<T*>(bsg_tile_group_remote_pointer(nxt_x,nxt_y,this->buffer));
      }

      __attribute__((always_inline))
      T *obtain_wr_ptr(){
        volatile unsigned int *o = &(this->occupancy_nxt[this->occ_idx]);
        bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (o)), 0);
        return this->buffer_remote + this->occ_idx * N;
      }

      __attribute__((always_inline))
      int finish_wr_ptr(){
        volatile unsigned int *o   = &(this->occupancy_nxt[this->occ_idx]);
        volatile unsigned int *o_r = this->occupancy_nxt_r + this->occ_idx;
        asm volatile("": : :"memory");
        *o   = 1;
        *o_r = 1;
        this->occ_idx = (this->occ_idx + 1) % DEPTH;
        return 0;
      }

      __attribute__((always_inline))
      T *obtain_rd_ptr(){
        volatile unsigned int *o = &(this->occupancy[this->occ_idx]);
        bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (o)), 1);
        return this->buffer + this->occ_idx * N;
      }

      __attribute__((always_inline))
      int finish_rd_ptr(){
        volatile unsigned int *o   = &(this->occupancy[this->occ_idx]);
        volatile unsigned int *o_r = this->occupancy_prv_r + this->occ_idx;
        asm volatile("": : :"memory");
        *o   = 0;
        *o_r = 0;
        this->occ_idx = (this->occ_idx + 1) % DEPTH;
        return 0;
      }

      __attribute__((always_inline))
      T* get_buffer(){
        return this->buffer;
      }
  };

}

#endif
