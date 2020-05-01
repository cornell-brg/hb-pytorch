#include <torch/csrc/python_headers.h>

#include <unordered_map>
#include <thread>
#include <chrono>
#include <sstream>
#include <TH/TH.h>
#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/HammerBladeGenerator.h>
#include <c10/hammerblade/HammerBladeFunctions.h>
// #include <c10/hammerblade/HammerBladeCachingAllocator.h>

#include <torch/csrc/hammerblade/THBP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/hammerblade_lazy_init.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/hammerblade/python_comm.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/Generator.h>

#ifdef HB_ENABLE_KERNEL_LOG
#include <c10/hammerblade/emul/kernel_logger.h>

extern KernelLogger kernel_call_logger;
#endif

// using namespace hammerblade;

////////////////////////////////////////////////////////////////////////////////
// HammerBlade management methods
////////////////////////////////////////////////////////////////////////////////


PyObject * THBPModule_getDevice_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  torch::utils::hammerblade_lazy_init();
  int device = c10::hammerblade::current_device();
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}


PyObject * THBPModule_set_run_yet_variable_to_false_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  torch::utils::hammerblade_set_run_yet_variable_to_false();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifdef HB_ENABLE_KERNEL_LOG
PyObject * THBPModule_enable_kernel_call_logger(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  kernel_call_logger.enable();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS;
}

PyObject * THBPModule_disable_kernel_call_logger(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  kernel_call_logger.disable();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// HammerBlade module initialization
////////////////////////////////////////////////////////////////////////////////

// Callback for python part. Used for additional initialization of python classes
static PyObject * THBPModule_initExtension(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  at::globalContext().lazyInitHammerBlade();

  auto m = THPObjectPtr(PyImport_ImportModule("torch.hammerblade"));
  if (!m) throw python_error();

  // // Register Storage Python objects with DynamicTypes.cpp
  // THBPDoubleStorage_postInit(m);
  // THBPFloatStorage_postInit(m);
  // THBPHalfStorage_postInit(m);
  // THBPLongStorage_postInit(m);
  // THBPIntStorage_postInit(m);
  // THBPShortStorage_postInit(m);
  // THBPCharStorage_postInit(m);
  // THBPByteStorage_postInit(m);
  // THBPBoolStorage_postInit(m);
  // THBPBFloat16Storage_postInit(m);

  bool has_half = false;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_half", has_half ? Py_True : Py_False);

  // auto num_gpus = c10::hammerblade::device_count();
  // auto default_opencl_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  // for(int i = 0; i < num_gpus; i++) {
  //   auto gen = at::opencl::detail::getDefaultOpenCLGenerator(i);
  //   auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
  //   // This reference is meant to be given away, so no need to incref here.
  //   PyTuple_SetItem(default_opencl_generators, i, (PyObject*)cast_gen);
  // }
  // set_module_attr("default_generators", default_opencl_generators);

  // bindOpenCLDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THBPModule_methods[] = {
  {"_hammerblade_init",        (PyCFunction)THBPModule_initExtension,    METH_NOARGS,  nullptr},
  {"_hammerblade_getDevice",   (PyCFunction)THBPModule_getDevice_wrap,   METH_NOARGS,  nullptr},
  {"_hammerblade_set_run_yet_variable_to_false",
    (PyCFunction)THBPModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
#ifdef HB_ENABLE_KERNEL_LOG
  {"_hammerblade_enable_kernel_call_logger",
    (PyCFunction)THBPModule_enable_kernel_call_logger, METH_NOARGS, nullptr},
  {"_hammerblade_disable_kernel_call_logger",
    (PyCFunction)THBPModule_enable_kernel_call_logger, METH_NOARGS, nullptr},
#endif
  {nullptr}
};

PyMethodDef* THBPModule_methods() {
  return _THBPModule_methods;
}

namespace torch { namespace hammerblade {

void initModule(PyObject *module) {
  python::initCommMethods(module);
}

}}
