//====================================================================
// sparse kernel common subroutine
// 08/14/2020 Zhonguan Zhao (zz546@zhang-21.ece.cornell.edu)
//====================================================================

#define NUM_OF_SLOTS 32
#define CACHE_LINE 8

inline int convert_idx(int index, int num_row, int row) {
  int idx = 0;
  int div = index / CACHE_LINE; 
  int mod = index % CACHE_LINE;
//  printf("div is %d\n", div);
//  printf("mod is %d\n", mod);
  if(num_row < NUM_OF_SLOTS) {
    int row_offset = row % num_row;
    idx = div * num_row * CACHE_LINE + mod + row_offset * CACHE_LINE;
  } else {
    int row_offset = row % NUM_OF_SLOTS;
    idx = div * NUM_OF_SLOTS * CACHE_LINE + mod + row_offset * CACHE_LINE;
  }
  return idx;
}
