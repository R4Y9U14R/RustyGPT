#ifndef CUDA_FUNCTIONS_CUH
#define CUDA_FUNCTIONS_CUH

#ifdef _WIN32
    #ifdef BUILD_DLL
        #define API_EXPORT __declspec(dllexport)
    #else
        #define API_EXPORT __declspec(dllimport)
    #endif
#else
    #define API_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <cstddef>

// ---------- DEBUGGING UTILS ----------
void print_matrix(const float* matrix, size_t num_rows, size_t num_cols);

// ---------- CUDA UTILS ----------
API_EXPORT bool check_is_cuda_available();

// ---------- CPU IMPLEMENTATIONS ----------
API_EXPORT void launch_add_cpu(const float* A, const float* B, float* result, size_t num_rows, size_t num_cols);
API_EXPORT void launch_matmul_cpu(const float* A, const float* B, float* result, size_t m_A, size_t n_A, size_t m_B, size_t n_B);
API_EXPORT void launch_transpose_cpu(const float* A, float* result, size_t num_rows, size_t num_cols);
API_EXPORT void launch_relu_cpu(const float* A, float* result, size_t num_rows, size_t num_cols);
API_EXPORT void launch_softmax_cpu(const float* logits, float* result, int n_classes);

// ---------- CUDA KERNELS ----------
__global__ void add_kernel(const float* A, const float* B, float* result, size_t num_rows, size_t num_cols);
__global__ void matmul_kernel(const float* A, const float* B, float* result, size_t m_A, size_t n_A, size_t m_B, size_t n_B);
__global__ void transpose_kernel(const float* A, float* result, size_t num_rows, size_t num_cols);
__global__ void relu_kernel(const float* A, float* result, size_t num_rows, size_t num_cols);
__global__ void softmax_kernel(const float* logits, float* result, int n_classes);

// ---------- CUDA LAUNCH FUNCTIONS ----------
API_EXPORT void launch_add(const float* A, const float* B, float* result, size_t num_rows, size_t num_cols);
API_EXPORT void launch_matmul(const float* A, const float* B, float* result, size_t m_A, size_t n_A, size_t m_B, size_t n_B);
API_EXPORT void launch_transpose(const float* A, float* result, size_t num_rows, size_t num_cols);
API_EXPORT void launch_relu(const float* A, float* result, size_t num_rows, size_t num_cols);
API_EXPORT void launch_softmax(const float* logits, float* result, int n_classes);

#ifdef __cplusplus
}
#endif

#endif // CUDA_FUNCTIONS_CUH
