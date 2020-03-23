//====================================================================
// An assert that works on both cosim and emul
// 03/16/2020 Lin Cheng (lc873@cornell.edu)
//====================================================================
#ifndef _BSG_ASSERT_HPP
#define _BSG_ASSERT_HPP

#define bsg_assert(cond) if (!(cond)) {                                          \
    bsg_printf("assert failed at %s:%d\n", __FILE__, __LINE__);                \
    bsg_fail();}

#endif
