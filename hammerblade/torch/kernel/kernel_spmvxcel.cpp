//==========================================================================================
//The kernel code of controlling the SpMV Xcels integrated to HammerBlade Manycore
//10/23/2020 Zhongyuan Zhao, Zichao Yue (zz546@cornell.edu)
//==========================================================================================

#include <kernel_common.hpp>

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
  CSR_RETURN
} CSRs;

extern "C" {

  __attribute__ ((noinline)) int tensorlib_spmvxcel(
    hb_tensor_t* _result,
    hb_tensor_t* _c2sr_m,
    hb_tensor_t* _vector,
    hb_tensor_t* _len_record,
    hb_tensor_t* _other_info) {

    auto result = (int*)HBTensor<int>(_result).data_ptr();
    auto c2sr_m = (int*)HBTensor<int>(_c2sr_m).data_ptr();
    auto vector = (int*)HBTensor<int>(_vector).data_ptr();
    auto len_record = (int*)HBTensor<int>(_len_record).data_ptr();
    auto other_info = (int*)HBTensor<int>(_other_info).data_ptr();

    int spmat_row = other_info[0];
    int spmat_col = other_info[1];
    int num_tile_x  = other_info[2];
    int last_tile_x = other_info[3];
    int num_tile_y = other_info[4];
    int last_tile_y = other_info[5];
    int tile_x_size = other_info[6];
    int tile_y_size = other_info[7];
    int num_pe = other_info[8];
    int cacheline_byte = other_info[9];
    int pe_region_log = other_info[10];
    int num_pe_log = other_info[11];
//    bsg_cuda_print_stat_kernel_start();
    if ((bsg_x == 0 ) && (bsg_y == 0)) {
      for(int i=0; i < 12; i++) {
        bsg_printf("other_info[%d] is %d\n", i, other_info[i]);
      }
      int record_len = num_tile_x * num_tile_y;
      int record_ptr[record_len];
      int total_len = 0;
      for (int i=0; i<record_len; i++) {
        record_ptr[i] = len_record[i];
        total_len += len_record[i];
      }
      bsg_printf("num_pe is %d\n", num_pe);
      int record_accumulate = 0;
      int pe_id;
      int init_index = ((unsigned)c2sr_m / cacheline_byte) % num_pe;
      
      for(int i = 0; i < num_tile_y; i++) {
        pe_id = init_index;
        bsg_printf("First pe_id is %d\n", pe_id);
        for(int pe_index = 0; pe_index < num_pe; pe_index++) {
          bsg_printf("The xcel with pe_id %d is working\n", pe_id);
          bsg_remote_int_ptr xcel_csr_base_ptr = bsg_global_ptr(30 + (pe_id % (num_pe/8)), 8 + (pe_id / (num_pe/8)), 0);
          pe_id = (pe_id + 1) % num_pe;
//          int out_ptr = (unsigned)result + pe_index * cacheline_byte + ((i * tile_y_size / num_pe * 4) / cacheline_byte) 
//                      * cacheline_byte * num_pe + ((i * tile_y_size / num_pe * 4) % cacheline_byte);
          int out_ptr = (int)result + (i * num_tile_y) * sizeof(int);
          int last_num_row = (last_tile_y % num_pe == 0) ? (last_tile_y / num_pe) : 
                             ((last_tile_y % num_pe > pe_index) ? (last_tile_y / num_pe + 1):(last_tile_y / num_pe));
          int num_row = (i == num_tile_y - 1)? last_num_row : (tile_y_size / num_pe);
          xcel_csr_base_ptr[CSR_PTR_B] = (unsigned)vector;
          xcel_csr_base_ptr[CSR_ADDR_SPACE_B] = num_pe_log;
          xcel_csr_base_ptr[CSR_ADDR_LOG] = pe_region_log;
          xcel_csr_base_ptr[CSR_INIT_ROW_ID] = pe_index;
          xcel_csr_base_ptr[CSR_NUM_ROW] = num_row;
          xcel_csr_base_ptr[CSR_PTR_OUT] = out_ptr;
          xcel_csr_base_ptr[CSR_NUM_TILE] = num_tile_x;
          xcel_csr_base_ptr[CSR_LAST_TILE_SIZE] = last_tile_x;
          xcel_csr_base_ptr[CSR_GO] = 1;
        }   
        
        for(int j = 0; j < num_tile_x; j++) {
          pe_id = init_index;
          for(int pe_index = 0; pe_index < num_pe; pe_index++) {
            bsg_remote_int_ptr xcel_csr_base_ptr = bsg_global_ptr(30 + (pe_id % (num_pe/8)), 8 + (pe_id / (num_pe/8)), 0);
            pe_id = (pe_id + 1) % num_pe;
            xcel_csr_base_ptr[CSR_PTR_A] = (unsigned)c2sr_m + record_accumulate; 
            xcel_csr_base_ptr[CSR_GO] = 1;
          }
          record_accumulate += record_ptr[i * num_tile_x + j];
        }       
      }
      
      pe_id = init_index;
      for(int pe_index = 0; pe_index < num_pe; pe_index++) {
        bsg_remote_int_ptr xcel_csr_base_ptr = bsg_global_ptr(30 + (pe_id % (num_pe/8)), 8 + (pe_id / (num_pe/8)), 0);
        pe_id = (pe_id + 1) % num_pe;
      }
    }
//    g_barrier.sync();
//    bsg_cuda_print_stat_kernel_end();
    return 0;
  }   
  HB_EMUL_REG_KERNEL(tensorlib_spmvxcel, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)
}    
