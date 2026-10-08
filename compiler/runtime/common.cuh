// NeuroCUDA runtime: shared helpers for every generated .cu file.
//
// Kept deliberately tiny: an error-checking macro for CUDA API calls and
// kernel launches, plus the elementwise/tile launch-configuration constants
// that src/cuda_codegen.py assumes when it computes grid/block dimensions.
// Generated code does `#include "common.cuh"` with runtime/ on the nvcc
// include path (`-I`), so this header is compiled once per generated
// program, not duplicated inline.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(_err));                 \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// Block size for elementwise kernels (add, relu, add_relu).
constexpr int NC_ELEMENTWISE_BLOCK = 16;

// Tile width for the shared-memory tiled matmul / matmul_relu kernels.
constexpr int NC_TILE = 32;
