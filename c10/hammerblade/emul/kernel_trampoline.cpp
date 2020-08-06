#include <kernel_trampoline.h>
#include <emul_hb_device.h>
#include <iostream>
#include <thread>

std::map<std::string, std::function<int(uint32_t, uint64_t*, uint32_t, uint32_t, uint32_t)>> kernelMap;
std::vector<std::function<int(uint32_t, uint64_t*, uint32_t, uint32_t, uint32_t)>> enqueued_kernel;
std::vector<uint32_t>  enqueued_argc;
std::vector<uint64_t*> enqueued_argv;

// HB device kernel logger
#ifdef HB_ENABLE_KERNEL_LOG
KernelLogger kernel_call_logger(false);
#endif

void enqueue_kernel(const std::string &kernel, uint32_t argc, uint64_t* argv) {
  assert (kernelMap.find(kernel) != kernelMap.end());
  enqueued_argc.push_back(argc);
  enqueued_argv.push_back(argv);
  enqueued_kernel.push_back(kernelMap[kernel]);
  if(const char* env_p = std::getenv("HBEMUL_DEBUG")) {
    std::cerr << "Emulation layer enqueued kernel " << kernel << std::endl;
  }
}

int execute_kernels() {
  if (enqueued_kernel.empty()) {
    return HB_MC_FAIL;
  }
  if (enqueued_argc.empty()) {
    return HB_MC_FAIL;
  }
  if (enqueued_argv.empty()) {
    return HB_MC_FAIL;
  }
  if (enqueued_kernel.size() != enqueued_argc.size()) {
    return HB_MC_FAIL;
  }
  if (enqueued_kernel.size() != enqueued_argv.size()) {
    return HB_MC_FAIL;
  }

  for (int i=0; i<enqueued_kernel.size(); i++) {
    std::vector<std::thread> tiles;
    for (size_t t = 0; t < (emul_hb_mesh_dim.x * emul_hb_mesh_dim.y); t++) {
      uint32_t idx = t / emul_hb_mesh_dim.y;
      uint32_t idy = t % emul_hb_mesh_dim.y;
      std::thread tile(enqueued_kernel[i], enqueued_argc[i], enqueued_argv[i],
                       idx, idy, t);
      tiles.push_back(std::move(tile));
    }
    if(const char* env_p = std::getenv("HBEMUL_DEBUG")) {
      std::cerr << "  Emulation layer launched " << tiles.size()
                << " threads to simulate the tile group"
                << std::endl;
    }
    for (auto& t : tiles) {
      t.join();
    }
  }

  while (!enqueued_kernel.empty()) {
    enqueued_kernel.pop_back();
    enqueued_argc.pop_back();
    enqueued_argv.pop_back();
  }

  return HB_MC_SUCCESS;
}
