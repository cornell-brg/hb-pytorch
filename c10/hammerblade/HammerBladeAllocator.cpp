#include <c10/hammerblade/HammerBladeAllocator.h>
#include <c10/core/DeviceType.h>

#include <bsg_manycore_cuda.h>
#include <bsg_manycore_errno.h>

namespace c10 {

void* alloc_hb(size_t nbytes) {
  hb_mc_eva_t data_p;
  int rc = hb_mc_device_malloc(&(hammerblade::_hb_device), (uint32_t) nbytes, &data_p);
  if (rc != HB_MC_SUCCESS) {
    AT_ERROR("HammerBlade: failed to allocate memory on device\n");
  }
  return (void *) ((intptr_t) data_p);
}

void free_hb(void* data) {
  hb_mc_eva_t data_p = (hb_mc_eva_t)((intptr_t)data);
  int rc = hb_mc_device_free(&(hammerblade::_hb_device), data_p);
  if (rc != HB_MC_SUCCESS) {
    AT_ERROR("HammerBlade: failed to free memory on device\n");
  }
}

/*
 * HammerBlade Allocator
 */
struct C10_API DefaultHammerBladeAllocator final : at::Allocator {
  DefaultHammerBladeAllocator() {}
  ~DefaultHammerBladeAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = alloc_hb(nbytes);
    return {data, data, &free_hb, at::Device(at::DeviceType::HAMMERBLADE)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &free_hb;
  }
};

void NoDelete(void*) {}

/*
 * HammerBlade Allocator
 */
at::Allocator* GetHammerBladeAllocator() {
  return GetAllocator(DeviceType::HAMMERBLADE);
}

/*
 * Global Default HammerBlade Allocator
 */
static DefaultHammerBladeAllocator g_hb_alloc;

/*
 * Get Default HammerBlade Allocator
 */
at::Allocator* GetDefaultHammerBladeAllocator() {
  return &g_hb_alloc;
}

REGISTER_ALLOCATOR(DeviceType::HAMMERBLADE, &g_hb_alloc);

} // namespace c10
