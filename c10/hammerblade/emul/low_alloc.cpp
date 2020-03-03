#include <low_alloc.h>
#include <cassert>
#include <iostream>
#include <sys/mman.h>

static size_t next_index = 0;
static size_t buffer_size = 1<<20;
static void* lowBuff = NULL;

void create_low_buffer() {
    lowBuff = mmap((void*)(32768), buffer_size,
                   PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    assert (lowBuff != NULL);
    assert ((uint64_t)((intptr_t)lowBuff) < UINT32_MAX);
    std::cerr << "Emulation low buffer allocated: " << lowBuff << " - " << (lowBuff + buffer_size) << std::endl;
    return;
}

void* low_malloc(size_t nbytes) {
    if (next_index + nbytes > buffer_size) {
        return NULL;
    }
    void* addr = lowBuff + next_index;
    next_index += nbytes;
    std::cerr << "Emulation low addr malloc at addr: " << addr << std::endl;
    return addr;
}

void low_free(void* addr) {
    return;
}
