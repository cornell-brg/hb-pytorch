#include <c10/hammerblade/HammerBladeAllocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/CPUAllocator.h>

namespace c10 {

void* alloc_hb(size_t nbytes) {
  return alloc_cpu(nbytes);
}

void free_hb(void* data) {
  free_cpu(data);
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
