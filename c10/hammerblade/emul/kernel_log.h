//=================================================================
// Kernel Logger
// 04/24/2020 Bandhav Veluri
//=================================================================

#ifndef _KERNEL_LOG_H_
#define _KERNEL_LOG_H_

#include <string>
#include <kernel_common.hpp>
#include <iostream>

// A popular C++ library for json pasrsing and
// serialization. Inlcuded as a header only library.
//
// Source: https://github.com/nlohmann/json
// Release: https://github.com/nlohmann/json/releases/tag/v3.7.3
#include <nlohmann/json.hpp>
using json = nlohmann::json;

/**
 * Implementation of Kernel Logging as JSON objects
 *
 * This class provides a single method interface to log arbitrary
 * kernel call thorugh `log_kernel_call` method. `log_kernel_call`
 * accepts variadic arguments and calls a private overloaded
 * routine capable of logging all possible argument types.
 */
class KernelLogger {
  private:
    // Runtime setting to enable or disable logging
    bool on;

    // Path to log file
    std::string log_path;

    // Json object for logging
    json log_json;

    // Kernel call currently being logged
    std::string curr_kernel;

  public:
    KernelLogger(bool on, std::string log_path) :
      on(on),
      log_path(log_path) {
        log_json = json();
        curr_kernel = "";
      }

    void enable() {
      on = true;
    }

    void disable() {
      on = false;
    }

    /**
     * Primary interface for using this class.
     *
     * This method recursively calls add_arg over each kernel
     * argument.
     */
    template<class T, class... Types>
    void log_kernel_call(T arg1, Types... args) {
      if(on) {
        add_arg(arg1);
        log_kernel_call(args...);
      }
    }

    void log_kernel_call() {
      // base case
      std::cout << log_json.dump(4) << std::endl;
      curr_kernel = "";
    }

  private:
    /**
     * Overloaded method to log all possible
     * kernel argument types
     */

    // Starts logging a kernel call
    void add_arg(const char* kernel) {
      log_json[kernel] = nullptr;
      curr_kernel = kernel;
    }

    // Logs kernel arguments
    void add_arg(hb_tensor_t*);
    void add_arg(hb_vector_t*);

    // Generic add_arg to handle standard types
    template<typename T>
    void add_arg(T* arg) {
      log_json[curr_kernel].push_back(*arg);
    }
};

#endif // _KERNEL_LOG_H_
