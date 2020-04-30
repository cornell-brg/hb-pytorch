//=================================================================
// Kernel Logger
// 04/24/2020 Bandhav Veluri
//=================================================================

#ifndef _KERNEL_LOG_H_
#define _KERNEL_LOG_H_

#ifndef HB_ENABLE_KERNEL_LOG
#error "kernel_logger.h included but HB_ENABLE_KERNEL_LOG not enabled!"
#endif

#include <ATen/ATen.h>
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
      if(on) {
        std::cout << std::endl << curr_kernel << std::endl;
        std::cout << log_json[curr_kernel].dump(4) << std::endl;
        curr_kernel = "";
      }
    }

  private:
    // Helper rotuine to convert C++ arry to a JSON list.
    template<typename T>
    json json_list(uint32_t N, T* data) {
      json obj = json::array();

      for(uint32_t i = 0; i < N; ++i) {
        obj.push_back(data[i]);
      }

      return obj;
    }

    /**
     * Overloaded method to log all possible
     * kernel argument types
     */

    // Starts logging a kernel call
    void add_arg(const char* kernel) {
      log_json[kernel] = nullptr;
      curr_kernel = kernel;
    }

    // Log tensor arguments
    void add_arg(hb_tensor_t* tensor) {
      uint32_t N = tensor->N;
      uint32_t dims = tensor->dims;
      uint32_t* strides = (uint32_t*) ((intptr_t) tensor->strides);
      uint32_t* sizes = (uint32_t*) ((intptr_t) tensor->sizes);
      uint64_t data_ptr = tensor->data;
      uint32_t storage_numel = tensor->storage_numel;

      json tensor_json;
      tensor_json["N"] = N;
      tensor_json["dims"] = dims;
      tensor_json["data_ptr"] = data_ptr;
      tensor_json["storage_head"] = (uint64_t) tensor->storage_head;
      tensor_json["strides"] = json_list(dims, strides);
      tensor_json["sizes"] = json_list(dims, sizes);
      tensor_json["storage"] = json_list(storage_numel, tensor->storage_head);
      log_json[curr_kernel].push_back(tensor_json);
    }

    // Log vector arguments
    void add_arg(hb_vector_t* vector) {
      json vector_json;
      vector_json["N"] = vector->N;
      vector_json["data"] = json_list(
          vector->N, (float*) ((intptr_t) vector->data));
      log_json[curr_kernel].push_back(vector_json);
    }

    void add_arg(void* arg) {
      // TODO: memcpy?
    }

    void add_arg(const void* arg) {
      // TODO: memcpy?
    }

    // Generic add_arg to handle standard types
    template<typename T>
    void add_arg(T* arg) {
      log_json[curr_kernel].push_back(*arg);
    }
};

#endif // _KERNEL_LOG_H_
