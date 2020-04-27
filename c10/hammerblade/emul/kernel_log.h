//=================================================================
// Kernel Logger
// 04/24/2020 Bandhav Veluri
//=================================================================

#ifndef _KERNEL_LOG_H_
#define _KERNEL_LOG_H_

#include <string>
#include <json_fwd.hpp>
#include <kernel_common.hpp>
#include <iostream>

using json = nlohmann::json;

/**
 * Implementation of Kernel Logging as JSON objects
 *
 * This class provides a single method interface to log arbitrary
 * kernel call thorugh `log_kernel_call` method. `log_kernel_call`
 * accepts variadic arguments and calls a private overloaded
 * routine capable of logging all possible argument types. As of
 * now, possible kernel argument types are:
 *
 * hb_tensor_t*
 * hb_vector_t*
 * float*
 * int32_t*
 * uint32_t*
 */
class KernelLogger {
  private:
    // Runtime setting to enable or disable logging
    bool on;

    // Path to log file
    std::string log_path;

    // Json object for logging
    json& log_json;

  public:
    KernelLogger(bool on, std::string log_path, json& log_json) :
      on(on),
      log_path(log_path),
      log_json(log_json) {}

    void enable() {
      on = true;
    }

    void disable() {
      on = false;
    }

    void log_kernel_call(const char* kernel) {
      if(on) {
        add_kernel(kernel);
      }
    }

    // Primary interface for using this class.
    //
    // This method recursively calls add_arg over each kernel
    // argument.
    template<class T, class... Types>
    void log_kernel_call(const char* kernel, Types... args, T argl) {
      if(on) {
        log_kernel_call(kernel, args...);
        add_arg(argl);
      }
    }

  private:
    // Starts logging a kernel call
    void add_kernel(const char*);

    // Overloaded method to log all possible
    // kernel argument types
    template<typename T>
    void add_arg(T arg) {
      std::cout << "  adding arg of type " << typeid(arg).name() << std::endl;
    }
};

#endif // _KERNEL_LOG_H_
