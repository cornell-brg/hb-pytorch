# Single-Tile Matrix Matrix Multiply

This example runs Vector-Vector Addition (A + B = C) on:
- `v0`: a single tile (1x1 grid of 1x1 tile group)

This example is intended to demonstrate how a single-tile vector add
can be optimized and how to use the tools.

The kernel code is located in the subdirectories of [kernel](kernel). The actual
vector-vector addition code is in the header file
[kernel/include/vector_add.hpp](kernel/include/vector_add.hpp). 

# Makefile Targets

For a list of all Makefile targets, run `make help`.

## Versions

There are several different versions of this kernel. Each is a subdirectory in
the [kernel](kernel) directory.

### Version 0

This is a vanilla vector addition implementation on a 1x1 grid of 1x1 tile group.
The entire addition is perfomed by a single tile, no work distribution is done.
Calls to bsg_print_stat_start/end are used to measure performance.