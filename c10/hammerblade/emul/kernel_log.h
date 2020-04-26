//=================================================================
// Kernel Logger
// 04/24/2020 Bandhav Veluri
//=================================================================

#ifndef _KERNEL_LOG_H_
#define _KERNEL_LOG_H_

#include <filesystem>
#include <json_fwd.hpp>
#include <kernel_common.hpp>

using fs = std::filesystem;
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
    fs::path log_path;

    // Json object for logging
    json log_json;

  public:
    KernelLogger(bool on, fs::path log_path) :
      on(on),
      log_path(log_path) {}

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
    void add_arg(hb_tensor_t* arg);
    void add_arg(hb_vector_t* arg);
    void add_arg(float* arg);
    void add_arg(int32_t* arg);
    void add_arg(uint32_t* arg);
}

#endif // _KERNEL_LOG_H
