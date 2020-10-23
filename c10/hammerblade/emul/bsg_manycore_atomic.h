#ifndef _BSG_MANYCORE_ATOMIC_H
#define _BSG_MANYCORE_ATOMIC_H

inline int bsg_amoswap (int* p, int val)
{
  int result;
  result = __sync_lock_test_and_set(p, val);
  return result;
}

inline int bsg_amoswap_aq (int* p, int val)
{
  return bsg_amoswap(p, val);
}

inline int bsg_amoswap_rl(int* p, int val)
{
  return bsg_amoswap(p, val);
}

inline int bsg_amoswap_aqrl(int* p, int val)
{
  return bsg_amoswap(p, val);
}


inline int bsg_amoor (int* p, int val)
{
  int result;
  result = __sync_fetch_and_or(p, val);
  return result;
}

inline int bsg_amoor_aq (int* p, int val)
{
  return bsg_amoor(p, val);
}

inline int bsg_amoor_rl (int* p, int val)
{
  return bsg_amoor(p, val);
}

inline int bsg_amoor_aqrl (int* p, int val)
{
  return bsg_amoor(p, val);
}


#endif
