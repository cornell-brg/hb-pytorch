#ifndef _HB_KERNEL_CGRA_COMMON_HPP
#define _HB_KERNEL_CGRA_COMMON_HPP

#define POD_CORD_X_WIDTH 3
#define POD_CORD_Y_WIDTH 4

#define TILE_CORD_X_WIDTH 4
#define TILE_CORD_Y_WIDTH 3

#define XCEL_X_CORD ((2) << TILE_CORD_X_WIDTH | (0))
#define XCEL_Y_CORD ((1) << TILE_CORD_Y_WIDTH | (0))

#define CONFIG_INST 0
#define LAUNCH_INST 1

#define CGRA_REG_CALC_GO          0
#define CGRA_REG_CFG_GO           1
#define CGRA_REG_ME_CFG_BASE_ADDR 2
#define CGRA_REG_PE_CFG_BASE_ADDR 3
#define CGRA_REG_PE_CFG_STRIDE    4
#define CGRA_REG_CFG_CMD          5
#define CGRA_REG_CFG_DONE         6
#define CGRA_REG_CALC_DONE        7

#define CGRA_BASE_SCRATCHPAD_WORD_ADDR 64

#define CGRA_REG_WR(ADDR, DATA)    \
  do {                                  \
    bsg_global_store(XCEL_X_CORD, XCEL_Y_CORD, ADDR << 2, DATA); \
  } while(0)

#define CGRA_REG_RD(ADDR, DATA)    \
  do {                                  \
    bsg_global_load(XCEL_X_CORD, XCEL_Y_CORD, ADDR << 2, DATA); \
  } while(0)

void stage_cfg(const unsigned int* src, int size) {
  int i;
  bsg_remote_int_ptr xcel_base_ptr = bsg_global_ptr(XCEL_X_CORD, XCEL_Y_CORD, 0);

  // The first 64 words are mapped to registers
  size = size-CGRA_BASE_SCRATCHPAD_WORD_ADDR;

  src += CGRA_BASE_SCRATCHPAD_WORD_ADDR;
  xcel_base_ptr += CGRA_BASE_SCRATCHPAD_WORD_ADDR;

  for (i = 0; i < size; i++)
    *(xcel_base_ptr + i) = src[i];
}

void execute_config(int ME_addr, int PE_addr, int PE_stride, int Cmd) {
  // 2
  CGRA_REG_WR(CGRA_REG_ME_CFG_BASE_ADDR, ME_addr);
  // 3
  CGRA_REG_WR(CGRA_REG_PE_CFG_BASE_ADDR, PE_addr);
  // 4
  CGRA_REG_WR(CGRA_REG_PE_CFG_STRIDE,    PE_stride);
  // 5
  CGRA_REG_WR(CGRA_REG_CFG_CMD,          Cmd);
  // 1
  CGRA_REG_WR(CGRA_REG_CFG_GO,           1);
}

void execute_launch() {
  CGRA_REG_WR(CGRA_REG_CALC_GO, 1);
}

#endif // _HB_KERNEL_CGRA_COMMON_HPP
