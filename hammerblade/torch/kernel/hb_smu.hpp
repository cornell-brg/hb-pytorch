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

// SMU padding encodings
#define PAD_W (0x1)
#define PAD_E (0x2)
#define PAD_N (0x4)
#define PAD_S (0x8)

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
               float* dst_base, int* ack,
               int block_dim_x, int block_dim_y ) {
  const uint32_t* src_strides = src.get_strides();
  float* src_addr = (float*)src.data_ptr() + (r_idx * block_dim_y * src_strides[0])
                    + (c_idx * block_dim_x * src_strides[1]);
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
              block_dim_x,                     // src_x_count
              src.get_strides()[0] * 4,        // src_y_stride
              block_dim_y,                     // src_y_count
              reinterpret_cast<int>(dst_base), // dst_base_addr
              reinterpret_cast<int>(ack),      // dst_ack_addr
              ((bsg_y+3) << 16) | (bsg_x+1)    // dst_cord
            );
}

inline void
launch_smu_conv_imap(
    size_t block_x, size_t block_y,
    size_t image_id, size_t filter_id, size_t channel_id,
    HBTensor<float, 4>& src,
    float* dst_base, int* ack,
    size_t block_dim_x, size_t block_dim_y,
    size_t filter_dim, size_t padding ) {

  // TODO: this needs to be fixed if filter_dim is not 3?
  const size_t imap_dim_x = block_dim_x + filter_dim - 1;
  const size_t imap_dim_y = block_dim_y + filter_dim - 1;

  size_t imap_x = block_x * block_dim_x;
  size_t imap_y = block_y * block_dim_y;
  // LC: this is used to correct the padding output offset
  // PP: this is correct and important. All imap blocks used for convolution
  // are not blocks without intersections. Instead, they share rows/cols of
  // values in their boundaries!
  imap_x = imap_x == 0 ? 0 : imap_x - padding;
  imap_y = imap_y == 0 ? 0 : imap_y - padding;

  size_t block_id = block_y * 2 + block_x;

  int pad = 0;

  const uint32_t* src_strides = src.get_strides();
  float* src_addr = (float*)src.data_ptr()
                  + (image_id * src_strides[0])
                  + (channel_id * src_strides[1])
                  + (imap_y * src_strides[2])
                  + (imap_x * src_strides[3]);

  // choose correct padding
  switch (block_id) {
    case 0:
      pad = PAD_N | PAD_W;
      src_addr = src_addr - padding - src_strides[2];
      break;
    case 1:
      pad = PAD_N | PAD_E;
      src_addr = src_addr - src_strides[2];
      break;
    case 2:
    case 4:
      pad = PAD_W;
      src_addr = src_addr - padding;
      break;
    case 3:
    case 5:
      pad = PAD_E;
      src_addr = src_addr;
      break;
    case 6:
      pad = PAD_W | PAD_S;
      src_addr = src_addr - padding;
      break;
    case 7:
      pad = PAD_E | PAD_S;
      src_addr = src_addr;
      break;
    default:
      pad = 0;
  }
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
  launch_smu( 0,                               // x
              bsg_y+3,                         // y
              pad,                             // padding
              reinterpret_cast<int>(src_addr), // src_base_addr
              src_strides[3] * 4,              // src_x_stride
              imap_dim_x,                      // src_x_count
              src_strides[2] * 4,              // src_y_stride
              imap_dim_y,                      // src_y_count
              reinterpret_cast<int>(dst_base), // dst_base_addr
              reinterpret_cast<int>(ack),      // dst_ack_addr
              ((bsg_y+3) << 16) | (bsg_x+1)    // dst_cord
            );

}

inline void
launch_smu_conv_grad(
    size_t block_x, size_t block_y,
    size_t image_id, size_t channel_id, size_t filter_id,
    HBTensor<float, 4>& src,
    float* dst_base, int* ack,
    size_t block_dim_x, size_t block_dim_y,
    size_t filter_dim, size_t padding ) {

  size_t grad_x = block_x * block_dim_x;
  size_t grad_y = block_y * block_dim_y;

  const uint32_t* src_strides = src.get_strides();
  float* src_addr = (float*)src.data_ptr()
                  + (image_id * src_strides[0])
                  + (channel_id * src_strides[1])
                  + (grad_y * src_strides[2])
                  + (grad_x * src_strides[3]);
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
  launch_smu( bsg_x+1,                         // x
              2,                               // y
              0,                               // padding
              reinterpret_cast<int>(src_addr), // src_base_addr
              4,                               // src_x_stride
              block_dim_x,                     // src_x_count
              src_strides[2] * 4,              // src_y_stride
              block_dim_y,                     // src_y_count
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
