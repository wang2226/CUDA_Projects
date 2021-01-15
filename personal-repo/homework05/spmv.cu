#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "spmv.h"
#include "math.h"

template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz,
                double* x, double* b)
{

    // COMPLETE THIS FUNCTION

    /*
    *  ELL Implementation with Shared Memory
    */


	extern __shared__ double cache[];

	// global thread index
    unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = thread_id / blockDim.x;

    if(row < m){
        double sum = 0.0;

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


    /*
    *  ELL Implementation Without Shared Memory
    */

/*
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
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters

    unsigned int blocks = m;
    unsigned int threads = 64;
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, m, n, nnz, x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}




void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n,
                      int nnz, double* x, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy ELL data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION

    CopyData(col_ind, m * n, sizeof(unsigned int), dev_col_ind);
    CopyData(vals, m * n, sizeof(double), dev_vals);
    CopyData(x, m, sizeof(double), dev_x);
    cudaMalloc((double**)dev_b, m * sizeof(double));

}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind,
                      double* vals, int m, int n, int nnz, double* x,
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy CSR data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION

    CopyData(row_ptr, m + 1, sizeof(unsigned int), dev_row_ptr);
    CopyData(col_ind, nnz, sizeof(unsigned int), dev_col_ind);
    CopyData(vals, nnz, sizeof(double), dev_vals);
    CopyData(x, n, sizeof(double), dev_x);
    cudaMalloc((double**)dev_b, m * sizeof(double));
}

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
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

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
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals,
            int m, int n, int nnz, double* x, double* b)
{
    // COMPLETE THIS FUNCTION

	   /*
	    *  CSR-Vector Implementation With Shared Memory
	    */


		extern __shared__ double cache[];

		// global thread index
	    unsigned int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	    unsigned int row = thread_id / blockDim.x;

	    if(row < m){
	        unsigned int row_begin = row_ptr[row];
	        unsigned int row_end = row_ptr[row + 1];

	        double sum = 0;
	        for(unsigned int index = row_begin + threadIdx.x; index < row_end; index += blockDim.x) {
	            // thread_id.x work many times within one row, and add together
	                sum += vals[index] * x[col_ind[index]];
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


	    /*
	    *  CSR-Scalar Implementation
	    */

/*
	    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

	    if(row < m){
	        unsigned int row_start = row_ptr[row];
	        unsigned int row_end   = row_ptr[row + 1];

	        double sum = 0.0;
	        for (unsigned int j = row_start; j < row_end; j++){
	            sum += vals[j] * x[col_ind[j]];
	        }
	        b[row] = sum;
	    }
*/

}


void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    // GPU execution parameters
    // 1 thread block per row
    // 64 threads working on the non-zeros on the same row

    unsigned int blocks = m;
    unsigned int threads = 64;
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, vals, m, n, nnz, x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}
