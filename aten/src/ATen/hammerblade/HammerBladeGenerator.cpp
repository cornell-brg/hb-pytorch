#include <ATen/HammerBladeGenerator.h>
#include <c10/util/C++17.h>
#include <algorithm>

namespace at {
namespace hammerblade {
namespace detail {

// Ensures default_gen_hb is initialized once.
static std::once_flag hammerblade_gen_init_flag;

// Default, global HammerBlade generator (seeder).
static std::shared_ptr<HammerBladeGenerator> default_gen_hb;

HammerBladeGenerator* getDefaultHammerBladeGenerator(DeviceIndex device_index) {
  TORCH_CHECK(device_index == -1); // we only support one HammerBlade device
  std::call_once(hammerblade_gen_init_flag, [&] {
    default_gen_hb = std::make_shared<HammerBladeGenerator>(at::detail::getNonDeterministicRandom());
  });
  return default_gen_hb.get();
}

}} // namespace hammerblade::detail

HammerBladeGenerator::HammerBladeGenerator(uint64_t seed_in)
  : Generator{Device(DeviceType::HAMMERBLADE)},
    engine_{seed_in} { }

void HammerBladeGenerator::set_current_seed(uint64_t seed) {
  engine_ = mt19937(seed);
}

uint64_t HammerBladeGenerator::current_seed() const {
  return engine_.seed();
}

uint64_t HammerBladeGenerator::seed() {
  auto random = detail::getNonDeterministicRandom();
  this->set_current_seed(random);
  return random;
}

DeviceType HammerBladeGenerator::device_type() {
  return DeviceType::HAMMERBLADE;
}

uint32_t HammerBladeGenerator::random() {
  return engine_();
}

at::mt19937 HammerBladeGenerator::engine() {
  return engine_;
}

void HammerBladeGenerator::set_engine(at::mt19937 engine) {
  engine_ = engine;
}

std::shared_ptr<HammerBladeGenerator> HammerBladeGenerator::clone() const {
  return std::shared_ptr<HammerBladeGenerator>(this->clone_impl());
}

HammerBladeGenerator* HammerBladeGenerator::clone_impl() const {
  auto gen = new HammerBladeGenerator();
  gen->set_engine(engine_);
  return gen;
}

} // namespace at
