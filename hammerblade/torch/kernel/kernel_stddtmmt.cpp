//====================================================================
// SampledT dense-denseT matrix multiply T
// 08/17/2020 Andrew Pareles (amp342@cornell.edu)
//====================================================================


#include <kernel_common.hpp>
#include <cmath>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_stddtmmt(
          hb_tensor_t* out_p, //destination
          hb_tensor_t* col_p, //cols
          hb_tensor_t* row_p, //rows
          hb_tensor_t* b_p, //dense
          hb_tensor_t* c_p //dense
          ) { 
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // SWAP ROW AND COL, since dealing with s.t
    auto cols = HBTensor<int>(row_p);
    auto rows = HBTensor<int>(col_p);
    auto b = HBTensor<float>(b_p);
    auto c = HBTensor<float>(c_p);
    auto res = HBTensor<float>(out_p);
    
    auto dp_len = b.dim(1); // i.e. b.size(1) or c.size(1)
    auto numel = cols.numel(); // i.e. cols.size() or rows.size()

    float sum;
    hb_tiled_for(numel, [&](size_t i) {
      int row = rows(i);
      int col = cols(i);
      sum = 0;
      for (int dot = 0; dot < dp_len; dot++){
        sum += b(row, dot) * c(col, dot);
      }
      res(col, row) = sum;
    });

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_stddtmmt, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}
