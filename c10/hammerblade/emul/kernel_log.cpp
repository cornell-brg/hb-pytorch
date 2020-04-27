#include <kernel_common.hpp>
#include <kernel_log.h>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Logs kernel arguments
void KernelLogger::add_arg(hb_tensor_t*) {}
void KernelLogger::add_arg(hb_vector_t*) {}
