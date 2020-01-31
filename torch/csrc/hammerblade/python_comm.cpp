#include <torch/csrc/utils/pybind.h>
// #include <torch/csrc/hammerblade/comm.h>
// #include <torch/csrc/hammerblade/utils.h>
// #include <torch/csrc/hammerblade/Stream.h>
#include <torch/csrc/hammerblade/THBP.h>
#include <torch/csrc/utils/auto_gil.h>
#include <ATen/core/functional.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <vector>

namespace torch { namespace hammerblade { namespace python {
void initCommMethods(PyObject *module) {
  return;
}
}}}
