//====================================================================
// Vector sort kernel
// 06/04/2020 Krithik Ranjan (kr397@cornell.edu)
//====================================================================

#include <kernel_common.hpp>

// Helper function to perform Quick Sort
// Modified implementation from 
// https://beginnersbook.com/2015/02/quicksort-program-in-c/
static void tensorlib_vsort_recur(
    HBTensor<float> * vec, 
    int32_t first, 
    int32_t last) {
        
    if (first >= last) {
        return;
    }
    else {
        int32_t i = first;
        int32_t j = last;
        int32_t piv = (first + last) / 2;

        while (i < j) {
            while ((*vec)(i) <= (*vec)(piv) && i < last) {
                i++;
            }
            while ((*vec)(j) > (*vec)(piv)) {
                j--;
            }

            if (i < j) {
                // Swap elements at i and j
                (*vec)(i) = (*vec)(i) + (*vec)(j);
                (*vec)(j) = (*vec)(i) - (*vec)(j);
                (*vec)(i) = (*vec)(i) - (*vec)(j);
            }
        }

        // Swap elements at j and piv
        (*vec)(piv) = (*vec)(piv) + (*vec)(j);
        (*vec)(j) = (*vec)(piv) - (*vec)(j);
        (*vec)(piv) = (*vec)(piv) - (*vec)(j);

        tensorlib_vsort_recur(vec, first, j-1);
        tensorlib_vsort_recur(vec, j+1, last);
    }


}

extern "C" {

  __attribute__ ((noinline))  int tensorlib_vsort(
          hb_tensor_t* result_p,
          hb_tensor_t* self_p) {

    // Convert low level pointers to Tensor objects
    HBTensor<float> result(result_p);
    HBTensor<float> self(self_p);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    // Use a single tile only
    if (__bsg_id == 0) {
      // Add 1 to each element
      for (size_t i = 0; i < self.numel(); i++) {
        result(i) = self(i);
      }

      tensorlib_vsort_recur(&result, 0, self.numel()-1);
    }

    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    // Sync
    g_barrier.sync();
    return 0;
  }

  // Register the HB kernel with emulation layer
  HB_EMUL_REG_KERNEL(tensorlib_vsort, hb_tensor_t*, hb_tensor_t*)

}
