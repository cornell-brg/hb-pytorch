#include <iostream>
#include <kernel_trampoline.h>

std::map<std::string, std::function<int(uint32_t, uint64_t*)>> kernelMap;
std::vector<std::function<int(uint32_t, uint64_t*)>> enqueued_kernel;
std::vector<uint32_t>  enqueued_argc;
std::vector<uint64_t*> enqueued_argv;

void enqueue_kernel(const std::string &kernel, uint32_t argc, uint64_t* argv) {
  assert (kernelMap.find(kernel) != kernelMap.end());
  enqueued_argc.push_back(argc);
  enqueued_argv.push_back(argv);
  enqueued_kernel.push_back(kernelMap[kernel]);
  std::cerr << "Emulation layer enqueued kernel " << kernel << std::endl;
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
    enqueued_kernel[i](enqueued_argc[i], enqueued_argv[i]);
  }

  while (!enqueued_kernel.empty()) {
    enqueued_kernel.pop_back();
    enqueued_argc.pop_back();
    enqueued_argv.pop_back();
  }

  return HB_MC_SUCCESS;
}
