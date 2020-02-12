#include "HammerBladeFunctions.h"

#include <mutex>

namespace c10 {
namespace hammerblade {

namespace {
static std::once_flag init_flag;
static void initHammerBladeDevice() {
  C10_HB_CHECK(hb_mc_device_init(&_hb_device, "HB_PYTORCH_PORT", 0));

  C10_HB_CHECK(hb_mc_device_program_init(&_hb_device, _bin_path, "default_allocator", 0));

  AT_WARN("HammerBlade Device Initialized\n");
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


void offload_kernel(const char* kernel, std::vector<eva_t> args) {

  uint32_t* cuda_argv = (uint32_t*) malloc(args.size() * sizeof(eva_t));
  if(!cuda_argv) {
    AT_ERROR("Falied to allocate cuda_argv!");
  }

  for(int i=0; i<args.size(); ++i) {
    cuda_argv[i] = args[i];
  }

  C10_HB_CHECK(hb_mc_kernel_enqueue(&_hb_device, _hb_grid_dim, _hb_tg_dim, kernel,
                                    args.size(), (const uint32_t*) cuda_argv));
  C10_HB_CHECK(hb_mc_device_tile_groups_execute(&_hb_device));

  free(cuda_argv);
}


}} // namespace c10::hammerblade
