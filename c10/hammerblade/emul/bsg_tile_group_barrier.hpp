//====================================================================
// bsg_tile_group_barrier.hpp
// 02/14/2019, shawnless.xie@gmail.com
// 02/19/2020, behsani@cs.washington.edu
//====================================================================
// The barrier implementation for tile group in manycore
// Usage:
//      1. #include "bsg_tile_group_barrier.hpp"
//      1. bsg_barrier<Y dimension, X dimension> my_barrier;
//      3. my_barrier.sync();
//
//Memory Overhead
//       (2 + X_DIM + 4) + (2 + Y_DIM +4)
//      =(12 + X_DIM + Y_DIM) BYTES
//
//Worst Case Performance:
//      1. row sync     :   X_DIM     cycles //all tiles writes to center tile of the row
//      2. row polling  : 3*X_DIM     cycles // lbu <xx>; beqz <xxx;
//      3. col sync     :   Y_DIM     cycles //all tiles writes to the center tiles of the col
//      4. col polling  : 3*Y_DIM     cycles // lbu <xx>; beqz <xx>;
//      5. col alert        Y_DIM     cycles // store
//      6. row alert        X_DIM     cycles // store
//      -----------------------------------------------
//                        5*( X_DIM + Y_DIM)
//      For 3x3 group,  cycles = 181, heavy looping/checking overhead for small groups.

#ifndef  BSG_TILE_GROUP_BARRIER_HPP_
#define  BSG_TILE_GROUP_BARRIER_HPP_

#include <cassert>
#include <cstddef>
#include <condition_variable>
#include <iostream>

// BARRIER_X_DIM and BARRIER_Y_DIM are *int* in the real impl
// But I changed them to typename so emulation layer can compile
// 05/21/2020, Lin Cheng (lc873@cornell.edu)
// native barrier adapted from
// https://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11

class bsg_barrier {
public:
  explicit bsg_barrier() :
    mThreshold(-1),
    mCount(-1),
    mGeneration(-1),
    initialized(false) {
  }

  void init(std::size_t iCount) {
    std::unique_lock<std::mutex> lLock{mMutex};
    assert(!initialized);
    mThreshold = iCount;
    mCount = iCount;
    mGeneration = 0;
    initialized = true;
    std::cerr << "Emulation barrier init'ed with " << mThreshold << " threads" << std::endl;
  }

  void sync() {
    std::unique_lock<std::mutex> lLock{mMutex};
    assert(initialized);
    auto lGen = mGeneration;
    if (!--mCount) {
        mGeneration++;
        mCount = mThreshold;
        mCond.notify_all();
    } else {
        mCond.wait(lLock, [this, lGen] { return lGen != mGeneration; });
    }
  }

private:
    std::mutex mMutex;
    std::condition_variable mCond;
    std::size_t mThreshold;
    std::size_t mCount;
    std::size_t mGeneration;
    bool initialized;
};
#endif
