#include <torch/csrc/utils/hammerblade_lazy_init.h>

#include <torch/csrc/python_headers.h>
#include <mutex>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch {
namespace utils {

static bool hb_run_yet = false;

void hammerblade_lazy_init() {
  AutoGIL g;
  if(!hb_run_yet) {
    auto module = THPObjectPtr(PyImport_ImportModule("torch.hammerblade"));
    if (!module) throw python_error();
    auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) throw python_error();
    hb_run_yet = true;
  }
}

void hammerblade_set_run_yet_variable_to_false() {
  hb_run_yet = false;
}

} // namespace utils
} // namespace torch
