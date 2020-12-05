//======================================================================
// Common operations related to SMU
//======================================================================

#ifndef _HB_SMU__
#define _HB_SMU__

// SMU registers
#define SMU_GO            (0  << 2)
#define SMU_PADDING       (2  << 2)
#define SMU_SRC_BASE_ADDR (3  << 2)
#define SMU_SRC_X_STRIDE  (4  << 2)
#define SMU_SRC_X_COUNT   (5  << 2)
#define SMU_SRC_Y_STRIDE  (6  << 2)
#define SMU_SRC_Y_COUNT   (7  << 2)
#define SMU_DST_BASE_ADDR (8  << 2)
#define SMU_DST_ACK_ADDR  (9  << 2)
#define SMU_DST_CORD      (10 << 2)

//----------------------------------------------------------------------
// Begin of helper functions
//----------------------------------------------------------------------

inline void
smu_register_wr( int x, int y, int addr, int value ) {
  bsg_global_store( x, y, addr, value );
}

//----------------------------------------------------------------------
// End of helper functions
//----------------------------------------------------------------------

inline void
launch_smu(int x, int y, int padding,
           int src_base_addr, int src_x_stride, int src_x_count,
           int src_y_stride, int src_y_count, int dst_base_addr,
           int dst_ack_addr, int dst_cord ) {
  smu_register_wr( x, y, SMU_PADDING,       padding       );
  smu_register_wr( x, y, SMU_SRC_BASE_ADDR, src_base_addr );
  smu_register_wr( x, y, SMU_SRC_X_STRIDE,  src_x_stride  );
  smu_register_wr( x, y, SMU_SRC_X_COUNT,   src_x_count   );
  smu_register_wr( x, y, SMU_SRC_Y_STRIDE,  src_y_stride  );
  smu_register_wr( x, y, SMU_SRC_Y_COUNT,   src_y_count   );
  smu_register_wr( x, y, SMU_DST_BASE_ADDR, dst_base_addr );
  smu_register_wr( x, y, SMU_DST_ACK_ADDR,  dst_ack_addr  );
  smu_register_wr( x, y, SMU_DST_CORD,      dst_cord      );
  smu_register_wr( x, y, SMU_GO,            1 );
}

inline void
launch_smu_mm( bool is_col, HBTensor<float, 2>& src, int r_idx, int c_idx,
               float* dst_base, int* ack ) {
  const uint32_t* src_strides = src.get_strides();
  float* src_addr = (float*)src.data_ptr() + (r_idx * BLOCK_DIM * src_strides[0])
                    + (c_idx * BLOCK_DIM * src_strides[1]);
  *ack = 0;
  /* bsg_printf("[INFO] Launching SMU with dst (%d, %d), r_idx = %d, c_idx = %d, " */
  /*            "stride0 = %d, stride1 = %d, " */
  /*            "src_base 0x%x, src_addr 0x%x, dst_base 0x%x, ack 0x%x (cleared), y_stride %u\n", */
  /*            bsg_y+3, bsg_x+1, r_idx, c_idx, */
  /*            src_strides[0], src_strides[1], */
  /*            reinterpret_cast<int>(src.data_ptr()), */
  /*            reinterpret_cast<int>(src_addr), */
  /*            reinterpret_cast<int>(dst_base), */
  /*            reinterpret_cast<int>(ack), */
  /*            src.get_strides()[0]*4); */
  launch_smu( is_col ? 0 : bsg_x+1,            // x
              is_col ? bsg_y+3 : 2,            // y
              0,                               // padding
              reinterpret_cast<int>(src_addr), // src_base_addr
              4,                               // src_x_stride
              BLOCK_DIM,                       // src_x_count
              src.get_strides()[0] * 4,        // src_y_stride
              BLOCK_DIM,                       // src_y_count
              reinterpret_cast<int>(dst_base), // dst_base_addr
              reinterpret_cast<int>(ack),      // dst_ack_addr
              ((bsg_y+3) << 16) | (bsg_x+1)    // dst_cord
            );
}

inline void wait_smu( int* ack ) {
  /* bsg_printf("[INFO] wait_smu: (%d, %d) waiting on 0x%x\n", bsg_y+3, bsg_x+1, ack ); */
  bsg_wait_local( ack, 1 );
}

#endif // _HB_SMU__
