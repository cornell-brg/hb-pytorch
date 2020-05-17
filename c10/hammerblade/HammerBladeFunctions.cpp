#include "HammerBladeFunctions.h"
#include <c10/probe/HBProfiler.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <mutex>
#include <string>

namespace c10 {
namespace hammerblade {

const int IDLE = -1;
const int IN_USE = -42;
std::atomic<int> hb_device_status{IDLE};

namespace {
static std::once_flag init_flag;
static void initHammerBladeDevice() {
  C10_HB_CHECK(hb_mc_device_init_custom_dimensions(&_hb_device, "HB_PYTORCH_PORT", 0, _hb_tg_dim));
  C10_HB_CHECK(hb_mc_device_program_init(&_hb_device, _bin_path, "default_allocator", 0));
  return;
}
} // namespace unnamed

DeviceIndex device_count() noexcept {
  // Assuming that we always have 1 HammerBlade manycore device
  int count = 1;
  // Lazy inialization
  std::call_once(init_flag, initHammerBladeDevice);
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  int cur_device = -1;
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  return; // no-op
}

 /* ------------------------------------------------------------------------------------
 * Interface to bsg_manycore runtime
 * -------------------------------------------------------------------------------------*/

eva_t device_malloc(size_t nbytes) {
  eva_t data_p;
  C10_HB_CHECK(hb_mc_device_malloc(&_hb_device, (uint32_t) nbytes, &data_p));
  return data_p;
}


void device_free(eva_t data_p) {
  C10_HB_CHECK(hb_mc_device_free(&_hb_device, data_p));
}


void* memcpy_host_to_device(void *dst, const void *src, uint32_t nbytes) {
  C10_HB_CHECK(hb_mc_device_memcpy(&_hb_device, dst, src, nbytes, HB_MC_MEMCPY_TO_DEVICE));
  return dst;
}


void* memcpy_device_to_host(void *dst, const void *src, uint32_t nbytes) {
  C10_HB_CHECK(hb_mc_device_memcpy(&_hb_device, dst, src, nbytes, HB_MC_MEMCPY_TO_HOST));
  return dst;
}


void* DMA_host_to_device(void *dst, const void *src, uint32_t nbytes) {
  hb_mc_dma_htod_t job = {.d_addr=(eva_t)((intptr_t)dst), .h_addr=src, .size=nbytes};
  C10_HB_CHECK(hb_mc_device_dma_to_device(&_hb_device, &job, 1));
  return dst;
}


void* DMA_device_to_host(void *dst, const void *src, uint32_t nbytes) {
  hb_mc_dma_dtoh_t job = {.d_addr=(eva_t)((intptr_t)src), .h_addr=dst, .size=nbytes};
  C10_HB_CHECK(hb_mc_device_dma_to_host(&_hb_device, &job, 1));
  return dst;
}


void offload_kernel(const char* kernel, std::vector<eva_t> args) {
  std::string kernel_str = "@OFFLOAD_KERNEL@__";

  int idle = IDLE;
  TORCH_CHECK(hb_device_status.compare_exchange_strong(idle, IN_USE),
      "HB device is already in use");

  kernel_str += kernel;
  c10::probe::LogATenKernelWithName(kernel_str);

  eva_t* cuda_argv = (eva_t*) malloc(args.size() * sizeof(eva_t));
  if(!cuda_argv) {
    AT_ERROR("Falied to allocate cuda_argv!");
  }

  for(int i=0; i<args.size(); ++i) {
    cuda_argv[i] = args[i];
  }

  // create a trim log so we can write logs directly
  c10::probe::HBProfilerTrimLog* trim_log = new c10::probe::HBProfilerTrimLog();

  C10_HB_CHECK(hb_mc_kernel_enqueue(&_hb_device, _hb_grid_dim, _hb_tg_dim, kernel,
                                    args.size(), cuda_argv));
  C10_HB_CHECK(hb_mc_device_tile_groups_execute(&_hb_device));

  // write the SIMULATED time to ExecutionTime log
  std::chrono::microseconds simulated(3154);
  trim_log->trim_manual_log_exec_time(simulated);
  // delete trim log
  delete trim_log;

  free(cuda_argv);

  int in_use = IN_USE;
  TORCH_CHECK(hb_device_status.compare_exchange_strong(in_use, IDLE),
      "HB device is not in use, how is this possible?");

}


}} // namespace c10::hammerblade
