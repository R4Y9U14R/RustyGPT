//
//  dense.cu
//
//  CUDA KERNELS FOR MATRIX OPERATIONS
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>

// ---------- ERROR HANDLING MACRO ----------

#define cudaCheck(err) {cudaCheckFunc((err), __FILE__, __LINE__);}
inline static void cudaCheckFunc(cudaError_t error, const char* file, int line, bool abort=true)
{
    if (error != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR!] at file " << file << ":" << line << std::endl
                  << cudaGetErrorString(error) << std::endl;
        if (abort)
            exit(error);
    }
}

// ---------- DEBUGGING UTILS ----------

void print_matrix(const float* matrix, const int num_rows, const int num_cols)
{
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            std::cout << matrix[i * num_cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// ---------- UTILITIES ----------

// Unused for now
static float* slice(const float* arr, const int start_idx, const int end_idx)
{
    if (start_idx < 0 || end_idx <= start_idx)
    {
        return nullptr;
    }

    int length = end_idx - start_idx;
    float* result_arr = (float*)malloc(length * sizeof(float));

    if (result_arr == nullptr)
    {
        return nullptr;
    }

    for (int i = start_idx; i < end_idx; ++i)
    {
        result_arr[i - start_idx] = arr[i];
    }

    return result_arr;
}

// ---------- CUDA UTILS ----------

extern "C" bool check_is_cuda_available()
{
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);
    if (error_id != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << " -> "  << cudaGetErrorString(error_id) << std::endl;
        return false;
    }
    return true;
}

// ---------- CPU IMPLEMENTATIONS ----------

extern "C" void launch_add_cpu(const float* A, const float* B, float* result, const int num_rows, const int num_cols)
{
    #pragma omp parallel for
    for (int i = 0; i < num_rows * num_cols; i++)
    {
        result[i] = A[i] + B[i];
    }
}

extern "C" void launch_matmul_cpu(const float* A, const float* B, float* result, const int m_A, const int n_A, const int m_B, const int n_B)
{
    #pragma omp parallel for collapse(2) num_threads(4)
    for (int i = 0; i < m_A; i++)
    {
        for (int j = 0; j < n_B; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < n_B; k++)
            {
                sum += A[i * n_A + k] * B[k * n_B + j];
            }
            result[i * n_B + j] = sum;
        }
    }
}

extern "C" void launch_transpose_cpu(const float* A, float* result, const int num_rows, const int num_cols)
{
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            result[i * num_cols + j] = A[j * num_rows + i];
        }
    }
}

// ---------- KERNELS ----------

__global__ void add_kernel(const float* A, const float* B, float* result, const int num_rows, const int num_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols)
    {
        int idx = row * num_cols + col;
        result[idx] = A[idx] + B[idx];
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* result, const int m_A, const int n_A, const int m_B, const int n_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m_A && col < m_B)
    {
        float sum = 0.0f;
        for (int k = 0; k < n_A; k++)
        {
            sum += A[row * n_A + k] * B[k * n_B + col];
        }
        result[row * n_B + col] = sum;
    }
}

__global__ void transpose_kernel(const float* A, float* result, const int num_rows, const int num_cols)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_rows && j < num_cols)
    {
        result[i * num_cols + j] = A[j * num_rows + i];
    }
}

// ---------- LAUNCH FUNCTIONS ----------

extern "C" void launch_add(const float* A, const float* B, float* result, const int num_rows, const int num_cols)
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

    print_matrix(h_result, num_rows, num_cols);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_result));
}

extern "C" void launch_matmul(const float* A, const float* B, float* result, const int m_A, const int n_A, const int m_B, const int n_B)
{
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

extern "C" void launch_transpose(const float* A, float* result, const int num_rows, const int num_cols)
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
