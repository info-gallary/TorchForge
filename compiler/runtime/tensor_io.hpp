// NeuroCUDA runtime: a tiny binary tensor file format.
//
// Layout: 4-byte magic "NCT1", int32 rows, int32 cols, then
// rows*cols float32 values in row-major order. Nothing fancier is
// needed -- this exists purely so that:
//   - generated programs can dump every `random(...)` input tensor and
//     every `print`ed output tensor to disk, and
//   - benchmarks/tensor_io.py can load those exact same arrays into
//     NumPy/PyTorch for a fair, same-data correctness and performance
//     comparison (see CLAUDE.md: "Compare against a PyTorch reference
//     using a stated numeric tolerance").
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

inline void nc_write_tensor_file(const char* path, const float* host_data, int rows, int cols) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "NeuroCUDA runtime: failed to open '%s' for writing\n", path);
        std::exit(1);
    }
    std::fwrite("NCT1", 1, 4, f);
    std::fwrite(&rows, sizeof(int), 1, f);
    std::fwrite(&cols, sizeof(int), 1, f);
    std::fwrite(host_data, sizeof(float), static_cast<size_t>(rows) * cols, f);
    std::fclose(f);
}

// Reads a .nct file written by nc_write_tensor_file (or by
// benchmarks/tensor_io.py's save_nct on the Python side) into a freshly
// malloc'd host buffer. Used by `load("path.nct")` NeuroLang programs to
// bring real external data (e.g. trained weights) into a tensor, instead
// of generating it with `random`/`zeros`. Caller must std::free() the
// returned pointer. Exits with a clear message on any mismatch, since a
// shape mismatch here would otherwise silently corrupt every later
// matmul/add in the program.
inline float* nc_read_tensor_file(const char* path, int expected_rows, int expected_cols) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "NeuroCUDA runtime: failed to open '%s' for reading\n", path);
        std::exit(1);
    }
    char magic[4];
    int rows = 0, cols = 0;
    std::fread(magic, 1, 4, f);
    std::fread(&rows, sizeof(int), 1, f);
    std::fread(&cols, sizeof(int), 1, f);
    if (std::memcmp(magic, "NCT1", 4) != 0) {
        std::fprintf(stderr, "NeuroCUDA runtime: '%s' is not a valid .nct tensor file\n", path);
        std::exit(1);
    }
    if (rows != expected_rows || cols != expected_cols) {
        std::fprintf(
            stderr,
            "NeuroCUDA runtime: '%s' has shape (%d, %d), but the program expects (%d, %d)\n",
            path, rows, cols, expected_rows, expected_cols);
        std::exit(1);
    }
    float* host_data = (float*)std::malloc(sizeof(float) * static_cast<size_t>(rows) * cols);
    std::fread(host_data, sizeof(float), static_cast<size_t>(rows) * cols, f);
    std::fclose(f);
    return host_data;
}
