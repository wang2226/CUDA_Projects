#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"



// -----------------------------------------------------------------
// For creating shared memory
template<class T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
// -----------------------------------------------------------------


// -----------------------------------------------------------------
// ELLPACK SPMV
template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz,
                double* x, double* b)
{
    // EXTRA CREDIT
	// ELL (row) with shared memory
    T *cache = SharedMemory<T>();

    // global thread index
    unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = thread_id / blockDim.x;

    if(row < m){
        T sum = 0;

        for(unsigned int index = threadIdx.x; index < n; index += blockDim.x) {
            unsigned int offset = row * n + index;
            if(vals[offset] != 0)
                sum += vals[offset] * x[col_ind[offset]];
        }
        cache[threadIdx.x] = sum;
        __syncthreads();

		// reduce local sums to row sum
        for(unsigned int s = blockDim.x / 2; s > 0; s /= 2){
            if(threadIdx.x < s){
                cache[threadIdx.x] += cache[threadIdx.x + s] ;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
            b[row] = cache[0];
    }

    /*
	//ELL (column) with shared memory
    T *cache = SharedMemory<T>();

	// global thread index
    unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = thread_id / blockDim.x;

    if(row < m){
        T sum = 0.0;

        for(unsigned int index = threadIdx.x; index < n; index += blockDim.x) {
            unsigned int offset = row + index * m;
            sum += vals[offset] * x[col_ind[offset]];
        }
        cache[threadIdx.x] = sum;
        __syncthreads();

		// reduce local sums to row sum
        for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
            if(threadIdx.x < stride){
                cache[threadIdx.x] += cache[threadIdx.x + stride] ;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
            b[row] = cache[0];
    }
    */

    /*
	//ELL (column) without shared memory
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        double  sum = 0.0;

        for (int index = 0; index < n; index++) {
            int offset = row + index * m;

            if (vals[offset] != 0.0) {
                sum += vals[offset] * x[col_ind[offset]];
            }
        }
        b[row] = sum;
    }
    */
}

void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz,
                  double* x, double* b)
{
    // GPU execution parameters
    unsigned int blocks = m;
    unsigned int threads = 64;
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, m, n,
                                                           nnz, x, b);
}


void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n,
                      int nnz, double* x, unsigned int** dev_col_ind,
                      double** dev_vals)
{
    // copy ELL data to GPU and allocate memory for output
    CopyData<unsigned int>(col_ind, m * n, sizeof(unsigned int), dev_col_ind);
    CopyData<double>(vals, m * n, sizeof(double), dev_vals);
}
// -----------------------------------------------------------------


// -----------------------------------------------------------------
// CSR SPMV
template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // find the start and end indicies for the target row
    unsigned int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(row_id < m) {
        unsigned int start = row_ptr[row_id];
        unsigned int end = row_ptr[row_id + 1];

        //  each thread calculates over non-zero element(s)
        T accum = 0.0;
        for(unsigned int i = start; i < end; i++) {
            accum += vals[i] * x[col_ind[i]];
        }
        // __syncthreads();
        b[row_id] = accum;
    }
}


void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
                int m, int n, int nnz, double* x, double* b)
{
    unsigned int threads = 256;
    unsigned int blocks = (m + threads - 1) / threads;
    unsigned int shared = 0;

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, vals,
                                                          m, n, nnz, x, b);
}


void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind,
                      double* vals, int m, int n, int nnz, double* x,
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals)
{
    // copy CSR data to GPU and allocate memory for output
    CopyData<unsigned int>(row_ptr, (m + 1), sizeof(unsigned int), dev_row_ptr);
    CopyData<unsigned int>(col_ind, nnz, sizeof(unsigned int), dev_col_ind);
    CopyData<double>(vals, nnz, sizeof(double), dev_vals);
}
// -----------------------------------------------------------------


// -----------------------------------------------------------------
void allocate_data_gpu(double* x, double* x_new, double* b, double* b_new, double** dx,
                       double** dx_new, double** db, double** db_new,
                       double** drk, double** dpk, double** dap, double** z1,
                       double** z2, int m, int n, int n_new)
{
    CopyData<double>(x, n, sizeof(double), dx);
    CopyData<double>(x_new, n, sizeof(double), dx_new);
    CopyData<double>(b, m, sizeof(double), db);
    CopyData<double>(b_new, m, sizeof(double), db_new);
    checkCudaErrors(cudaMalloc((void**) drk, sizeof(double) * m));
    checkCudaErrors(cudaMalloc((void**) dpk, sizeof(double) * m));
    checkCudaErrors(cudaMalloc((void**) dap, sizeof(double) * m));
    checkCudaErrors(cudaMemset((void*) *drk, 0, sizeof(double) * m));
    checkCudaErrors(cudaMemset((void*) *dpk, 0, sizeof(double) * m));
    checkCudaErrors(cudaMemset((void*) *dap, 0, sizeof(double) * m));

    int next_p2 = n;
    if(!((n != 0) && ((n & (n - 1)) == 0))) {
       next_p2 = pow(2, (int) log2((double) n) + 1);
    }
    checkCudaErrors(cudaMalloc((void**) z1, sizeof(double) * next_p2));
    checkCudaErrors(cudaMalloc((void**) z2, sizeof(double) * next_p2));
    checkCudaErrors(cudaMemset((void*) *z1, 0, sizeof(double) * next_p2));
    checkCudaErrors(cudaMemset((void*) *z2, 0, sizeof(double) * next_p2));
}
// -----------------------------------------------------------------


// -----------------------------------------------------------------
void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    // printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
    //      (m * sizeof(double)) * 1e-6 / elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
// -----------------------------------------------------------------


// -----------------------------------------------------------------
template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  // printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
  //        (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
// -----------------------------------------------------------------



// -----------------------------------------------------------------
// Other GPU Kernels
template <class T>
__global__ void
vec_add_kernel(T c, T* x, T* y, T* z, int n)
{
    // COMPLETE THIS
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = c * x[i] + y[i];
    }
}


void vec_add_gpu(const int n, const double a, double* x, double* y, double* z)
{
    unsigned int threads = 256;
    unsigned int blocks = (n + threads - 1) / threads;
    unsigned int shared = 0;
    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    vec_add_kernel<double><<<dimGrid, dimBlock, shared>>>(a, x, y, z, n);
}
// -----------------------------------------------------------------


// -----------------------------------------------------------------
// Kernels needed for dnrm2
template <class T>
__global__ void
reduce_kernel(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // COMPLETE THIS
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Global index
    int local_id = threadIdx.x; // Local thread ID

    sdata[local_id] = g_idata[thread_id] +g_idata[thread_id + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            sdata[local_id] += sdata[local_id + s];
        }
        __syncthreads ();
    }
    if (local_id == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}


template <class T>
__global__ void
vec_mul_kernel(T c, T* x, T* y, T* z, int n)
{
    // COMPLETE THIS
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // Global index
    for (int i = thread_id; i < n; i += blockDim.x * gridDim.x) {
        z[i] = c * x[i] * y[i];
    }
}


double ddot_gpu(int n, double* x, double *y, double* z1, double* z2)
{
    // Create temporary buffer
    int next_p2 = n;
    if(!((n != 0) && ((n & (n - 1)) == 0))) {
       next_p2 = pow(2, (int) log2((double) n) + 1);
    }
    checkCudaErrors(cudaMemset((void*) z1, 0, sizeof(double) * next_p2));
    checkCudaErrors(cudaMemset((void*) z2, 0, sizeof(double) * next_p2));

    // COMPLETE THIS
    // first do a * b
    // then reduce
    unsigned int threads = 256;
    unsigned int blocks = (n + threads - 1) / threads;
    unsigned int shared = threads * sizeof(double);
    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    vec_mul_kernel<double><<<dimGrid, dimBlock, shared>>>(1, x, y, z1, n);

    reduce_kernel<double><<<dimGrid, dimBlock, shared>>>(z1, z2, n);

    double *z3 = (double *)malloc(blocks * sizeof(double));
    get_result_gpu(z2, z3, blocks);

    // return answer
    double dot = 0;
    for(int i = 0; i < blocks; i++)
    {
        dot += z3[i];
    }

    free(z3);
    return dot;
}


double dnrm2_gpu(const int n, double* x, double* z1, double* z2)
{
    double nrm = ddot_gpu(n, x, x, z1, z2);
    return sqrt(nrm);
}
// -----------------------------------------------------------------


// -----------------------------------------------------------------
int cg_gpu_csr(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
               double *x, double* b, double* rk, double* pk, double* ap,
               double* z1, double* z2, int m, int n, int nnz, int max_iter,
               double tol)
{
    // r0 = b - Ax
    spmv_gpu(row_ptr, col_ind, vals, m, n, nnz, x, ap);
    vec_add_gpu(m, -1.0, ap, b, rk);

    // if r0 is sufficiently small, return x0 as the result
    double residual = dnrm2_gpu(n, rk, z1, z2);
    if(residual < tol) {
        fprintf(stdout, "\tInput is the solution\n");
        return 0;
    } else {
        fprintf(stdout, "\n\tInitial residual is %f\n", residual);
    }

    // p0 = r0
    checkCudaErrors(cudaMemcpy(pk, rk, sizeof(double) * m,
                               cudaMemcpyDeviceToDevice));

    int k = 0;
    double residual_new = 0.0;
    // repeat until convergence of max iterations has been reached
    for(int i = 0; i < max_iter; i++) {
        // A * p
        spmv_gpu(row_ptr, col_ind, vals, m, n, nnz, pk, ap);
        // d = p^T * A * p
        double dotprod = ddot_gpu(m, pk, ap, z1, z2);
        // alpha = (r^t * r) / d;
        double alpha = (residual * residual) / dotprod;

        // xk = xk + alpha * pk
        vec_add_gpu(m, alpha, pk, x, x);
        // rk = rk - alpha * A*p
        vec_add_gpu(m, (-1.0 * alpha), ap, rk, rk);

        // r^t * r
        residual_new = dnrm2_gpu(m, rk, z1, z2);
        if(residual_new < tol) {
            fprintf(stdout, "\tSolution calculated. Final residual: %f\n",
                    residual_new);
            break;
        } else {
        //    fprintf(stdout, "\tIt: %d\tresidual: %f\n", k, residual_new);
        }

        // beta = (r^t * r) / residual
        double beta = (residual_new * residual_new) / (residual * residual);

        // p = r + beta * p
        vec_add_gpu(m, beta, pk, rk, pk);

        residual = residual_new;
        k++;
    }
    return 0;
}
// -----------------------------------------------------------------
int cg_gpu_ell(unsigned int* col_ind, double* vals,
               double *x, double* b, double* rk, double* pk, double* ap,
               double* z1, double* z2, int m, int n, int nnz, int max_iter,
               double tol)
{
    checkCudaErrors(cudaMemset((void*) rk, 0, sizeof(double) * m));
    checkCudaErrors(cudaMemset((void*) pk, 0, sizeof(double) * m));
    checkCudaErrors(cudaMemset((void*) ap, 0, sizeof(double) * m));
    // r0 = b - Ax
    spmv_gpu_ell(col_ind, vals, m, n, nnz, x, ap);
    vec_add_gpu(m, -1.0, ap, b, rk);

    // if r0 is sufficiently small, return x0 as the result
    double residual = dnrm2_gpu(m, rk, z1, z2);
    if(residual < tol) {
        fprintf(stdout, "\tInput is the solution\n");
        return 0;
    } else {
        fprintf(stdout, "\n\tInitial residual is %f\n", residual);
    }

    // p0 = r0
    checkCudaErrors(cudaMemcpy(pk, rk, sizeof(double) * m,
                               cudaMemcpyDeviceToDevice));

    int k = 0;
    double residual_new = 0.0;
    // repeat until convergence of max iterations has been reached
    for(int i = 0; i < max_iter; i++) {
        // A * p
        spmv_gpu_ell(col_ind, vals, m, n, nnz, pk, ap);
        // d = p^T * A * p
        double dotprod = ddot_gpu(m, pk, ap, z1, z2);
        // alpha = (r^t * r) / d;
        double alpha = (residual * residual) / dotprod;

        // xk = xk + alpha * pk
        vec_add_gpu(m, alpha, pk, x, x);
        // rk = rk - alpha * A*p
        vec_add_gpu(m, (-1.0 * alpha), ap, rk, rk);

        // r^t * r
        residual_new = dnrm2_gpu(m, rk, z1, z2);
        if(residual_new < tol) {
            fprintf(stdout, "\tSolution calculated. Final residual: %f\n",
                    residual_new);
            break;
        } else {
             // fprintf(stdout, "\tIt: %d\tresidual: %f\n", k, residual_new);
        }

        // beta = (r^t * r) / residual
        double beta = (residual_new * residual_new) / (residual * residual);

        // p = r + beta * p
        vec_add_gpu(m, beta, pk, rk, pk);

        residual = residual_new;
        k++;
    }
    return 0;
}


// -----------------------------------------------------------------
// Free GPU memory
void free_gpu(unsigned int* drp, unsigned int* dci, unsigned int* dec,
              double* dev, double* dx, double* dx_new, double* db, double* db_new,
              double* drk, double* dpk,
              double* dap, double* z1, double* z2)
{
    checkCudaErrors(cudaFree(drp));
    checkCudaErrors(cudaFree(dci));
    checkCudaErrors(cudaFree(dec));
    checkCudaErrors(cudaFree(dev));
    checkCudaErrors(cudaFree(dx));
    checkCudaErrors(cudaFree(dx_new));
    checkCudaErrors(cudaFree(db));
    checkCudaErrors(cudaFree(db_new));
    checkCudaErrors(cudaFree(drk));
    checkCudaErrors(cudaFree(dpk));
    checkCudaErrors(cudaFree(dap));
    checkCudaErrors(cudaFree(z1));
    checkCudaErrors(cudaFree(z2));
}
// -----------------------------------------------------------------
