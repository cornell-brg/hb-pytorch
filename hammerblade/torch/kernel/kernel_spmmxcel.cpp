#include <kernel_common.hpp>
#include <kernel_sparse_common.hpp>

enum {
  CSR_GO = 0, // go/fetch result
  CSR_NUM_HBM ,
  CSR_PE_PER_HBM ,
  CSR_NUM_ROW ,
  CSR_PTR_OUT ,
  CSR_PTR_A ,
  CSR_PTR_B ,
  CSR_NUM_PE,
  CSR_INIT_ROW_ID,
  CSR_ADDR_SPACE_B ,
  CSR_BURST_MAX ,
  CSR_ADDR_LOG ,
  CSR_NUM_TILE ,
  CSR_LAST_TILE_SIZE,
  CSR_OPERATION_MODE,
  CSR_VECTOR_SIZE,
  CSR_RETURN
} CSRs;

extern "C" {

  __attribute__ ((noinline)) int tensorlib_spmmxcel(
    hb_tensor_t* _result,
    hb_tensor_t* _c2sr_m,
    hb_tensor_t* _matrix,
    hb_tensor_t* _other_info,
    int32_t* _n,
    int32_t* _k) {

    int* result = (int*)HBTensor<int>(_result).data_ptr();
    int* c2sr_m = (int*)HBTensor<int>(_c2sr_m).data_ptr();
    int* matrix = (int*)HBTensor<int>(_matrix).data_ptr();
    int* other_info = (int*)HBTensor<int>(_other_info).data_ptr();

    int m = other_info[0];
    int n = *(_n);
    int k = *(_k);
    
    int pe_region_log = other_info[3];
    int num_pe_log = other_info[4];
    if((__bsg_x == 0 ) && (__bsg_y == 0)) {
      bsg_printf("\nManycore>> Hello from core %d, %d in group origin=(%d,%d).\n", __bsg_x, __bsg_y, __bsg_grp_org_x, __bsg_grp_org_y);
      bsg_printf("m, n and k are %d, %d and %d\n", m, n, k);
      int num_pe_test = NUM_PE;
      int pe_id;
      unsigned init_index = ((unsigned)c2sr_m / CACHELINE_BYTE) % NUM_PE;
      bsg_printf("init_index is %d\n", init_index);
      pe_id = init_index;
      for(unsigned pe_index = 0; pe_index < num_pe_test; pe_index++) {
        bsg_remote_int_ptr xcel_csr_base_ptr = bsg_global_ptr(30 + (pe_id%(NUM_PE/8)), (8 + (pe_id/ (NUM_PE/8))), 0);
//        bsg_remote_int_ptr xcel_csr_base_ptr = bsg_global_ptr(14, 2, 0);
        pe_id = (pe_id + 1) % NUM_PE;
        int out_ptr = (int)result + pe_index * CACHELINE_BYTE;
        bsg_printf("address of result is %u, %d and %u\n", (unsigned)result, (int)result, &(result[0]));   
        int num_row = (pe_index < m % NUM_PE) ? (m + NUM_PE - 1) / NUM_PE : m / NUM_PE;
        xcel_csr_base_ptr[CSR_PTR_B] = (int)matrix;
        bsg_printf("address of matrix is %u, %d and %u\n", (unsigned)matrix, (int)matrix, &(matrix[0]));
        xcel_csr_base_ptr[CSR_PTR_A] = (int)c2sr_m;
        bsg_printf("address of c2sr_m is %u, %d and %u\n", (unsigned)c2sr_m, (int)c2sr_m, &(c2sr_m[0]));
        xcel_csr_base_ptr[CSR_ADDR_SPACE_B] = num_pe_log; 
        xcel_csr_base_ptr[CSR_ADDR_LOG] = pe_region_log;
        xcel_csr_base_ptr[CSR_INIT_ROW_ID] = pe_index;
        xcel_csr_base_ptr[CSR_NUM_ROW] = num_row;
        xcel_csr_base_ptr[CSR_PTR_OUT] = out_ptr;
        xcel_csr_base_ptr[CSR_OPERATION_MODE] = 1;
        xcel_csr_base_ptr[CSR_VECTOR_SIZE] = k;
        xcel_csr_base_ptr[CSR_GO] = 1;
      }
     
      bsg_printf("Finish loop 1 !!!!!!!!!!!!!!!!!!!!!!!!\n");
      pe_id = init_index;

      for(int pe_index = 0; pe_index < num_pe_test; pe_index++) {
        bsg_remote_int_ptr xcel_csr_base_ptr = bsg_global_ptr(30 + (pe_id%(NUM_PE/8)), (8 + (pe_id/(NUM_PE/8))), 0);
//        bsg_remote_int_ptr xcel_csr_base_ptr = bsg_global_ptr(14, 2, 0);
        int temp = xcel_csr_base_ptr[CSR_RETURN];
        pe_id = (pe_id + 1) % NUM_PE;
      }
      for(unsigned tk =0; tk<100; tk++){
        bsg_printf(" ");
      }  
      bsg_printf("===== Execution finished! ===== \n"); 
    }
    return 0;
  }
  HB_EMUL_REG_KERNEL(tensorlib_spmmxcel, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, int32_t*, int32_t*)
}

