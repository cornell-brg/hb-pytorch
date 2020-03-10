#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/MT19937RNGEngine.h>

namespace at {

struct CAFFE2_API HammerBladeGenerator : public Generator {
  // Constructors
  HammerBladeGenerator(uint64_t seed_in = 42);
  ~HammerBladeGenerator() = default;

  // HammerBladeGenerator methods
  std::shared_ptr<HammerBladeGenerator> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  static DeviceType device_type();
  uint32_t random();
  at::mt19937 engine();
  void set_engine(at::mt19937 engine);

private:
  HammerBladeGenerator* clone_impl() const override;
  at::mt19937 engine_;
};

namespace hammerblade {
namespace detail {

CAFFE2_API HammerBladeGenerator* getDefaultHammerBladeGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace hammerblade
} // namespace at
