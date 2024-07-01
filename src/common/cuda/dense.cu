//
//  dense.cu
//
//  CUDA KERNELS FOR MATRIX OPERATIONS
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <omp.h>
#include <cmath>

// ---------- ERROR HANDLING MACRO ----------

#define cudaCheck(err) {__cudaCheck((err), __FILE__, __LINE__);}
inline static void __cudaCheck(cudaError_t error, const char* file, int line, bool abort=true)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "[CUDA ERROR!] at file %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        if (abort)
            exit(error);
    }
}

// ---------- DEBUGGING UTILS ----------

void print_matrix(const float* matrix, size_t num_rows, size_t num_cols)
{
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            printf("%d ", matrix[i * num_cols + j]);
        }
        printf("\n");
    }
}

// ---------- CUDA UTILS ----------

extern "C" bool check_is_cuda_available()
{
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);
    if (error_id != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount returned %d -> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
        return false;
    }
    return true;
}

// ---------- CPU IMPLEMENTATIONS ----------

extern "C" void launch_add_cpu(const float* A, const float* B, float* result, size_t num_rows, size_t num_cols)
{
    #pragma omp parallel for
    for (int i = 0; i < num_rows * num_cols; i++)
    {
        result[i] = A[i] + B[i];
    }
}

extern "C" void launch_matmul_cpu(const float* A, const float* B, float* result, size_t m_A, size_t n_A, size_t m_B, size_t n_B)
{
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < m_A; i++)
    {
        for (int j = 0; j < n_B; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < n_A; k++)
            {
                sum += A[i * n_A + k] * B[k * n_B + j];
            }
            result[i * n_B + j] = sum;
        }
    }
}

extern "C" void launch_transpose_cpu(const float* A, float* result, size_t num_rows, size_t num_cols)
{
    #pragma omp for
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            result[i * num_cols + j] = A[j * num_rows + i];
        }
    }
}

extern "C" void launch_relu_cpu(const float* A, float* result, size_t num_rows, size_t num_cols)
{
    #pragma omp for
    for (size_t i =  0; i < num_cols; i++)
    {
        for (size_t j = 0; j < num_cols; j++)
        {
            int idx = i * num_cols + j;
            result[idx] = A[idx] > 0 ? A[idx] : 0.0f;
        }
    }
}

extern "C" void launch_softmax_cpu(const float* logits, float* result, int n_classes)
{
    float denominator = 0.0f;
    for (int j = 0; j < n_classes; j++)
    {
        denominator += std::exp(logits[j]);
    }
    for (int j = 0; j < n_classes; j++)
    {
        float numerator = std::exp(logits[j]);
        result[j] = numerator / denominator;
    }
}

// ---------- KERNELS ----------

__global__ void add_kernel(const float* A, const float* B, float* result, size_t num_rows, size_t num_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols)
    {
        int idx = row * num_cols + col;
        result[idx] = A[idx] + B[idx];
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* result, size_t m_A, size_t n_A, size_t m_B, size_t n_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m_A && col < n_B)
    {
        float sum = 0.0f;
        for (int k = 0; k < n_A; k++)
        {
            sum += A[row * n_A + k] * B[k * n_B + col];
        }
        result[row * n_B + col] = sum;
    }
}

__global__ void transpose_kernel(const float* A, float* result, size_t num_rows, size_t num_cols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_rows && j < num_cols)
    {
        result[i * num_cols + j] = A[j * num_rows + i];
    }
}

__global__ void relu_kernel(const float* A, float* result, size_t num_rows, size_t num_cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows * num_cols)
    {
        result[i] = result[i] = A[i] > 0 ? A[i] : 0;
    }
}

__global__ void softmax_kernel(const float* logits, float* result, int n_classes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_classes)
    {
        float denominator = 0.0f;
        for (int j = 0; j < n_classes; j++)
        {
            denominator += std::exp(logits[j]);
        }
        float numerator = std::exp(logits[i]);
        result[i] = numerator / denominator;
    }
}

// ---------- LAUNCH FUNCTIONS ----------

extern "C" void launch_add(const float* A, const float* B, float* result, size_t num_rows, size_t num_cols)
{
    size_t size = (num_rows * num_cols) * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_result = (float*)malloc(size);

    memcpy(h_A, A, size);
    memcpy(h_B, B, size);
    memcpy(h_result, result, size);

    float* d_A;
    cudaCheck(cudaMalloc(&d_A, size));
    float* d_B;
    cudaCheck(cudaMalloc(&d_B, size));
    float* d_result;
    cudaCheck(cudaMalloc(&d_result, size));

    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    add_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_result, num_rows, num_cols);
    
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));

    memcpy(result, h_result, size);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_result));
}

extern "C" void launch_matmul(const float* A, const float* B, float* result, size_t m_A, size_t n_A, size_t m_B, size_t n_B)
{
    if (n_A != m_B)
    {
        fprintf(stderr, "Matricies of shape (%zu, %zu) and (%zu, %zu) cannot be multiplied together.\n", m_A, n_A, m_B, n_B);
        exit(EXIT_FAILURE);
    }

    size_t size_A = (m_A * n_A) * sizeof(float);
    size_t size_B = (m_B * n_B) * sizeof(float);
    size_t size_result = (m_A * n_B) * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_result = (float*)malloc(size_result);

    memcpy(h_A, A, size_A);
    memcpy(h_B, B, size_B);
    memcpy(h_result, result, size_result);

    float* d_A ;
    cudaCheck(cudaMalloc(&d_A, size_A));
    float* d_B;
    cudaCheck(cudaMalloc(&d_B, size_B));
    float* d_result;
    cudaCheck(cudaMalloc(&d_result, size_result));

    cudaCheck(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n_B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m_A + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_result, m_A, n_A, m_B, n_B);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(h_result, d_result, size_result, cudaMemcpyDeviceToHost));

    memcpy(result, h_result, size_result);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_result));
}

extern "C" void launch_transpose(const float* A, float* result, size_t num_rows, size_t num_cols)
{
    size_t size = (num_rows * num_cols) * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_result = (float*)malloc(size);

    memcpy(h_A, A, size);
    memcpy(h_result, result, size);

    float* d_A;
    cudaCheck(cudaMalloc(&d_A, size));
    float* d_result;
    cudaCheck(cudaMalloc(&d_result, size));

    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_result, num_rows, num_cols);
    
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));

    memcpy(result, h_result, size);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_result));
}

extern "C" void launch_relu(const float* logits, float* result, size_t num_rows, size_t num_cols)
{
    size_t size = (num_rows * num_cols) * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_result = (float*)malloc(size);

    memcpy(h_A, logits, size);
    memcpy(h_result, result, size);

    float* d_A;
    cudaCheck(cudaMalloc(&d_A, size));
    float* d_result;
    cudaCheck(cudaMalloc(&d_result, size));

    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    relu_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_result, num_rows, num_cols);
    
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));

    memcpy(result, h_result, size);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_result));
}

extern "C" void launch_softmax(const float* logits, float* result, int n_classes)
{
    size_t size = n_classes * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_result = (float*)malloc(size);

    memcpy(h_A, logits, size);
    memcpy(h_result, result, size);

    float* d_A;
    cudaCheck(cudaMalloc(&d_A, size));
    float* d_result;
    cudaCheck(cudaMalloc(&d_result, size));

    cudaCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n_classes + threadsPerBlock.x - 1) / threadsPerBlock.x);

    softmax_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_result, n_classes);
    
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));

    memcpy(result, h_result, size);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_result));
}
