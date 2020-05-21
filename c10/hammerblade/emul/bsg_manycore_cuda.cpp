// Copyright (c) 2019, University of Washington All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
// 
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// 
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <bsg_manycore_cuda.h>
#include <bsg_manycore_errno.h>
#include <kernel_trampoline.h>
#include <emul_hb_device.h>

#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#define DEBUG 0

// print out a warning to stderr to remind us that we are
// running emulation layer
#define EMUL_WARNING() if(DEBUG) \
  fprintf(stderr, "Emulating function %s\n", __PRETTY_FUNCTION__);

// global variables for states
bool device_busy = false;
bool binary_loaded = false;

// reset global state so testing is easier
void reset_runtime() {
  device_busy = false;
  binary_loaded = false;
}

namespace {
static size_t get_env_num_tiles(const char* var_name, size_t def_value = 0) {
  try {
    if (auto* value = std::getenv(var_name)) {
      int nthreads = std::stoi(value);
      assert(nthreads > 0);
      return nthreads;
    }
  } catch (const std::exception& e) {
    std::cerr << "Invalid " << var_name << " variable value, " << e.what() << std::endl;
  }
  return def_value;
}
}

// FAKE implemenations

        /**
         * BRG CUDALite Emulation
         * If we haven't already call device_init, return HB_MC_SUCCESS
         * Otherwise, return HB_MC_INITIALIZED_TWICE
         */
        /**
         * Initializes the manycor struct, and a mesh structure with default (maximum)
         * diemsnions inside device struct with list of all tiles and their cooridnates 
         * @param[in]  device        Pointer to device
         * @param[in]  name          Device name
         * @param[in]  id            Device id
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_init (hb_mc_device_t *device,
                               const char *name,
                               hb_mc_manycore_id_t id) {
          EMUL_WARNING();
          return hb_mc_device_init_custom_dimensions(device, name, id, {0,0});
        }



        /**
         * BRG CUDALite Emulation
         * We dont support init with custom dimensions
         * So return HB_MC_FAIL
         */
        /**
         * Initializes the manycor struct, and a mesh structure with custom
         * diemsnions inside device struct with list of all tiles and their cooridnates 
         * @param[in]  device        Pointer to device
         * @param[in]  name          Device name
         * @param[in]  id            Device id
         * @param[in]  dim           Tile pool (mesh) dimensions
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_init_custom_dimensions (hb_mc_device_t *device,
                                                 const char *name,
                                                 hb_mc_manycore_id_t id,
                                                 hb_mc_dimension_t dim) {
          fprintf(stderr, "Emulating CUDALite...\n");
          EMUL_WARNING();
          if (id != 0) {
            return HB_MC_INVALID;
          }
          if (!device_busy) {
            // mark device as busey
            device_busy = true;
            // return default or customized size
            device->mesh = (hb_mc_mesh_t *) malloc (sizeof (hb_mc_mesh_t));
            if (device->mesh == NULL) {
              return HB_MC_NOMEM;
            }
            // If the input dimensions are (0,0) this will initialize the whole array.
            // In emulation, we read device size from ENV
            if (!dim.x && !dim.y) {
              dim.x = get_env_num_tiles("HBEMUL_TILE_X_DIM", 1);;
              dim.y = get_env_num_tiles("HBEMUL_TILE_Y_DIM", 1);;
            }
            device->mesh->dim = dim;
            emul_hb_mesh_dim = dim;
            return HB_MC_SUCCESS;
          } else {
            return HB_MC_INITIALIZED_TWICE;
          }
        }




        /**
         * BRG CUDALite Emulation
         * We dont support initing from a buffer
         * So return HB_MC_FAIL
         */
        /**
         * Takes in a buffer containing the binary and its size,
         * freezes tiles, loads program binary into all tiles and into dram,
         * and sets the symbols and registers for each tile.
         * @param[in]  device        Pointer to device
         * @parma[in]  bin_name      Name of binary elf file
         * @param[in]  bin_data      Buffer containing binary 
         * @param[in]  bin_size      Size of the binary to be loaded onto device
         * @param[in]  id            Id of program's memory allocator
         * @param[in]  alloc_name    Unique name of program's memory allocator
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_program_init_binary (hb_mc_device_t *device,
                                              const char *bin_name,
                                              const unsigned char *bin_data,
                                              size_t bin_size,
                                              const char *alloc_name,
                                              hb_mc_allocator_id_t id) {
          EMUL_WARNING();
          return HB_MC_FAIL;
        }





        /**
         * BRG CUDALite Emulation
         * We dont care what is the bin_name since we compile the functions
         * into the emulation runtime
         * But we only allow default_allocator
         */
        /**
         * Takes in a binary name, loads the binary from file onto a buffer,
         * freezes tiles, loads program binary into all tiles and into dram,
         * and sets the symbols and registers for each tile.
         * @param[in]  device        Pointer to device
         * @parma[in]  bin_name      Name of binary elf file
         * @param[in]  id            Id of program's memory allocator
         * @param[in]  alloc_name    Unique name of program's memory allocator
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_program_init (hb_mc_device_t *device,
                                       const char *bin_name,
                                       const char *alloc_name,
                                       hb_mc_allocator_id_t id) {
          EMUL_WARNING();
          if (!device_busy) {
            return HB_MC_UNINITIALIZED;
          }
          const char *default_allocator = "default_allocator";
          if (strcmp(alloc_name, default_allocator) != 0) {
            return HB_MC_INVALID;
          }
          if (id != 0) {
            return HB_MC_INVALID;
          }
          if (!binary_loaded) {
            binary_loaded = true;

            return HB_MC_SUCCESS;
          } else {
            return HB_MC_INITIALIZED_TWICE;
          }
        }



        /**
         * Allocates memory on device DRAM
         * hb_mc_device_program_init() or hb_mc_device_program_init_binary() should
         * have been called before calling this function to set up a memory allocator.
         * @param[in]  device        Pointer to device
         * @parma[in]  size          Size of requested memory
         * @param[out] eva           Eva address of the allocated memory
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_malloc (hb_mc_device_t *device, uint32_t size, hb_mc_eva_t *eva) {
          EMUL_WARNING();
          if (!device_busy || !binary_loaded) {
            return HB_MC_UNINITIALIZED;
          }
          void *alloc = malloc(size);
          if (alloc == NULL) {
            return HB_MC_FAIL;
          } else {
            *eva = (hb_mc_eva_t)((intptr_t) alloc);
            return HB_MC_SUCCESS;
          }
        }





        /**
         * Copies a buffer from src on the host/device DRAM to dst on device DRAM/host.
         * @param[in]  device        Pointer to device
         * @parma[in]  src           EVA address of source to be copied from
         * @parma[in]  dst           EVA address of destination to be copied into
         * @param[in]  name          EVA address of dst
         * @param[in]  count         Size of buffer (number of words) to be copied
         * @param[in]  hb_mc_memcpy_kind         Direction of copy (HB_MC_MEMCPY_TO_DEVICE / HB_MC_MEMCPY_TO_HOST)
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_memcpy (hb_mc_device_t *device,
                                 void *dst,
                                 const void *src,
                                 uint32_t count,
                                 enum hb_mc_memcpy_kind kind) {
          EMUL_WARNING();
          if (!device_busy || !binary_loaded) {
            return HB_MC_UNINITIALIZED;
          }
          memcpy(dst, src, (size_t)count);
          return HB_MC_SUCCESS;
        }





        /**
         * Sets memory to a give value starting from an address in device's DRAM.
         * @param[in]  device        Pointer to device
         * @parma[in]  eva           EVA address of destination 
         * @param[in]  val           Value to be written out
         * @param[in]  sz            The number of bytes to write into device DRAM
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_memset (hb_mc_device_t *device,
                                 const hb_mc_eva_t *eva,
                                 uint8_t data,
                                 size_t sz) {
          EMUL_WARNING();
          if (!device_busy || !binary_loaded) {
            return HB_MC_UNINITIALIZED;
          }
          void* dst = (void*)((intptr_t)eva);
          memset(dst, data, sz);
          return HB_MC_SUCCESS;
        }





        /**
         * Frees memory on device DRAM
         * hb_mc_device_program_init() or hb_mc_device_program_init_binary() should
         * have been called before calling this function to set up a memory allocator.
         * @param[in]  device        Pointer to device
         * @param[out] eva           Eva address of the memory to be freed
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_free (hb_mc_device_t *device, hb_mc_eva_t eva) {
          EMUL_WARNING();
          if (!device_busy || !binary_loaded) {
            return HB_MC_UNINITIALIZED;
          }
          void* alloc = (void*)((intptr_t)eva);
          free(alloc);
          return HB_MC_SUCCESS;
        }





        /**
         * Enqueues and schedules a kernel to be run on device
         * Takes the grid size, tile group dimensions, kernel name, argc,
         * argv* and the finish signal address, calls hb_mc_tile_group_enqueue
         * to initialize all tile groups for grid.
         * @param[in]  device        Pointer to device
         * @param[in]  grid_dim      X/Y dimensions of the grid to be initialized
         * @param[in]  tg_dim        X/Y dimensions of tile groups in grid
         * @param[in]  name          Kernel name to be executed on tile groups in grid
         * @param[in]  argc          Number of input arguments to kernel
         * @param[in]  argv          List of input arguments to kernel
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_kernel_enqueue (hb_mc_device_t *device,
                                       hb_mc_dimension_t grid_dim,
                                       hb_mc_dimension_t tg_dim,
                                       const char *name,
                                       const uint32_t argc,
                                       uint64_t *argv) {
          EMUL_WARNING();
          if (!device_busy || !binary_loaded) {
            return HB_MC_UNINITIALIZED;
          }
          // assume a single grid for now
          // pytorch tile group size should match emulation tile group size
          if (grid_dim.x * grid_dim.y != 1
              || tg_dim.x != emul_hb_mesh_dim.x
              || tg_dim.y != emul_hb_mesh_dim.y) {
            return HB_MC_FAIL;
          }
          std::string _name(name);
          enqueue_kernel(_name, argc, argv);
          return HB_MC_SUCCESS;
        }




        /**
         * Iterates over all tile groups inside device,
         * allocates those that fit in mesh and launches them. 
         * API remains in this function until all tile groups
         * have successfully finished execution.
         * @param[in]  device        Pointer to device
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_tile_groups_execute (hb_mc_device_t *device) {
          EMUL_WARNING();
          if (!device_busy || !binary_loaded) {
            return HB_MC_UNINITIALIZED;
          }
          return execute_kernels();
        }





        /**
         * Deletes memory manager, device and manycore struct, and freezes all tiles in device.
         * @param[in]  device        Pointer to device
         * @return HB_MC_SUCCESS if succesful. Otherwise an error code is returned.
         */
        int hb_mc_device_finish (hb_mc_device_t *device) {
          EMUL_WARNING();
          if (!device_busy || !binary_loaded) {
            return HB_MC_UNINITIALIZED;
          }
          device_busy = false;
          binary_loaded = false;
          return HB_MC_SUCCESS;
        }





        /**
         * Copy data using DMA from the host to the device.
         * @param[in] device  Pointer to device
         * @param[in] jobs    Vector of host-to-device DMA jobs
         * @param[in] count   Number of host-to-device jobs
         */
        int hb_mc_device_dma_to_device (hb_mc_device_t *device,
                                        const hb_mc_dma_htod_t *jobs,
                                        size_t count)
        {
          EMUL_WARNING();
          int err;
          // for each job...
          for (size_t i = 0; i < count; i++) {
            const hb_mc_dma_htod_t *dma = &jobs[i];
            err = hb_mc_device_memcpy
                    (device,
                     (void*)((intptr_t)dma->d_addr),
                     dma->h_addr,
                     dma->size,
                     HB_MC_MEMCPY_TO_DEVICE);

            if (err != HB_MC_SUCCESS) {
              return err;
            }
          }
          return HB_MC_SUCCESS;
        }




        /**
         * Copy data using DMA from the device to the host.
         * @param[in] device  Pointer to device
         * @param[in] jobs    Vector of device-to-host DMA jobs
         * @param[in] count   Number of device-to-host jobs
         */
        int hb_mc_device_dma_to_host(hb_mc_device_t *device,
                                     const hb_mc_dma_dtoh_t *jobs,
                                     size_t count)
        {
          EMUL_WARNING();
          int err;
          // for each job...
          for (size_t i = 0; i < count; i++) {
            const hb_mc_dma_dtoh_t *dma = &jobs[i];
            err = hb_mc_device_memcpy
                    (device,
                     dma->h_addr,
                     (void*)((intptr_t)dma->d_addr),
                     dma->size,
                     HB_MC_MEMCPY_TO_HOST);

            if (err != HB_MC_SUCCESS) {
              return err;
            }
          }
          return HB_MC_SUCCESS;
        }
