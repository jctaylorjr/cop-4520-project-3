#include <assert.h>
#include <iostream>
#include <limits.h>
#include "timing.cuh"

#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, ANSI_COLOR_RED "GPUassert: %s %s %d\n" ANSI_COLOR_RESET, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//data generator
void dataGenerator(int *data, int count, int first, int step)
{
	assert(data != NULL);

	for (int i = 0; i < count; ++i) {
		data[i] = RAND_RANGE(INT_MAX);
	}
	srand(time(NULL));
	for (int i = count - 1; i > 0; i--) //knuth shuffle
	{
		int j = RAND_RANGE(i);
		int k_tmp = data[i];
		data[i] = data[j];
		data[j] = k_tmp;
	}
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
	uint bits;
	asm("bfe.u32 %0, %1, %2, %3;"
	    : "=r"(bits)
	    : "r"(x), "r"(start), "r"(nbits));
	return bits;
}

__device__ unsigned int int_to_int(unsigned int k) {
    return (k == 0 || k == 1 ? k : ((k % 2) + 10 * int_to_int(k / 2)));
}

//define the histogram kernel here
__global__ void histogram(int *d_data, int *d_hist, int n_partitions, int n_elements)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	//uses grid stride
	while (i < n_elements) {
		int h = bfe(d_data[i], 31 - (uint)log2f(n_partitions),
			    (uint)log2f(n_partitions));
		//printf("Thread ID: %d\th: %d\td_data[i]: %d\tbinary:%d\n", i, h,
		//      d_data[i], int_to_int(h));
		atomicAdd(&d_hist[h], 1);
		i += stride;
	}
}

// define the prefix scan kernel here
// implement it yourself or borrow the code from CUDA samples
// ---------------------------------------------------------
// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call, then
// uniformly adding across the input data via the uniform_add<<<>>> kernel.
__global__ void prefixScan(int *data, int n_partitions, int *partial_sums = NULL)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	//uses grid stride
	while (i < n_partitions) {
		if (i == 0) {
			partial_sums[i] = 0;
		} else {
			partial_sums[i] = data[i - 1] + partial_sums[i - 1];
		}
		i += stride;
	}
}

//define the reorder kernel here
__global__ void Reorder(int *d_data, int *d_hist, int n_partitions, int n_elements, int *d_partial_sums, int *d_ordered)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	//uses grid stride
	while (i < n_elements) {
		int h = bfe(d_data[i], 31 - (uint)log2f(n_partitions),
			    (uint)log2f(n_partitions));
		offset = atomicAdd(&d_partial_sums[h], 1);
		d_ordered[offset] = d_data[i];
		printf("offset: %d\td_data[i]: %d\tbinary: %d\n", offset, d_data[i], int_to_int(h));
		i += stride;
	}
}

__global__ void uniform_add(int *data, int *partial_sums, int len)
{
	__shared__ int buf;
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (id > len)
		return;

	if (threadIdx.x == 0) {
		buf = partial_sums[blockIdx.x];
	}

	__syncthreads();
	data[id] += buf;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
	return ((dividend % divisor) == 0) ? (dividend / divisor) :
						   (dividend / divisor + 1);
}

int main(int argc, char *argv[])
{
	int cuda_device = 0;
	cudaDeviceProp deviceProp;
	gpuErrchk(cudaGetDeviceProperties(&deviceProp, cuda_device));

	int n_elements = 1000000;
	int n_partitions = 1024;

	if (argc > 1) {
		n_elements = atoi(argv[1]);
	}
	if (argc > 2) {
		n_partitions = atoi(argv[2]);
	}

	int *h_data, *h_hist, *h_partial_sums, *h_result, *h_ordered;
	int *d_data, *d_hist, *d_partial_sums, *d_ordered;
	int sz = sizeof(int) * n_elements;

	gpuErrchk(cudaMallocHost((void **)&h_data, sizeof(int) * n_elements));
	gpuErrchk(cudaMallocHost((void **)&h_result, sizeof(int) * n_elements));

	dataGenerator(h_data, n_elements, 0, 1);

	int blockSize = 256;
	int gridSize = ceil((double)n_partitions / blockSize);
	int nWarps = ceil((double)blockSize / 32);
	int shmem_sz = nWarps * sizeof(int);
	int n_partialSums = n_partitions;
	//int n_partialSums = ceil((float)n_elements / blockSize);
	int partial_sz = n_partialSums * sizeof(int);
	int p_blockSize = min(n_partialSums, blockSize);
	int p_gridSize = iDivUp(n_partialSums, p_blockSize);
	//iDivUp(n_partialSums, p_blockSize)

	printf("Scan summation for %d elements, %d partial sums\n", n_elements,
	       n_partialSums);

	printf("Partial summing %d elements with %d blocks of size %d\n",
	       n_partialSums, p_gridSize, p_blockSize);

	gpuErrchk(cudaMalloc((void **)&d_data, sz));
	gpuErrchk(cudaMalloc((void **)&d_partial_sums, partial_sz));
	gpuErrchk(cudaMemset(d_partial_sums, 0, partial_sz));

	gpuErrchk(cudaMallocHost((void **)&h_partial_sums, partial_sz));
	gpuErrchk(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));

	// initialize a timer
	cudaEvent_t start, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	float et = 0;
	float inc = 0;

	gpuErrchk(cudaMalloc((void **)&d_data, sz));
	gpuErrchk(cudaMalloc((void **)&d_partial_sums, partial_sz));
	gpuErrchk(cudaMemset(d_partial_sums, 0, partial_sz));

	gpuErrchk(cudaMallocHost((void **)&h_partial_sums, partial_sz));
	gpuErrchk(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMallocHost((void **)&h_hist, n_partitions * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_hist, n_partitions * sizeof(int)));

	gpuErrchk(cudaMallocHost((void **)&h_ordered, n_elements * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_ordered, n_elements * sizeof(int)));

	gpuErrchk(cudaEventRecord(start, 0));

	histogram<<<gridSize, blockSize>>>(d_data, d_hist, n_partitions, n_elements);
	cudaDeviceSynchronize();

	gpuErrchk(cudaMemcpy(h_hist, d_hist, sizeof(int) * n_partitions, cudaMemcpyDeviceToHost));
	
	prefixScan<<<gridSize, blockSize, shmem_sz>>>(d_hist, n_partitions,
							  d_partial_sums);
	cudaDeviceSynchronize();
	/*

	prefixScan<<<p_gridSize, p_blockSize, shmem_sz>>>(d_partial_sums,
							      32);
	cudaDeviceSynchronize();

	uniform_add<<<gridSize - 1, blockSize>>>(d_data + blockSize,
						 d_partial_sums, n_elements);
	cudaDeviceSynchronize();
	*/

	Reorder<<<gridSize, blockSize>>>(d_data, d_hist, n_partitions, n_elements, d_partial_sums, d_ordered);
	cudaDeviceSynchronize();

	gpuErrchk(cudaEventRecord(stop, 0));
	gpuErrchk(cudaEventSynchronize(stop));
	gpuErrchk(cudaEventElapsedTime(&inc, start, stop));
	et += inc;

	gpuErrchk(cudaMemcpy(h_result, d_data, sz, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_partial_sums, d_partial_sums, partial_sz,
			     cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_ordered, d_ordered, n_elements * sizeof(int),
	cudaMemcpyDeviceToHost));

	int i = 0;
	while(i < n_partitions) {
		printf("partition %d:\toffset: %d\tnumber of keys: %d.\n", i, h_partial_sums[i] -= h_hist[i], h_hist[i]);
		i++;
	}
	/*
	printf("\nNUMBER OF PARTITIONS: %d\n", n_partitions);
	i = 0;
	while(i < n_partialSums) {
		printf("Test Sum %d: %d\n", i, h_partial_sums[i]);
		i++;
	}
	*/

	i = 0;
	while(i < n_elements) {
		printf("h_ordered %d: %d\n", i, h_ordered[i]);
		i++;
	}
	//printf("Test Sum: %d\n", h_partial_sums[n_partialSums - 1]);
	printf("Time (ms): %f\n", et);
	printf("%d elements scanned in %f ms -> %f MegaElements/s\n",
	       n_elements, et, n_elements / (et / 1000.0f) / 1000000.0f);

	gpuErrchk(cudaFreeHost(h_data));
	gpuErrchk(cudaFreeHost(h_result));
	gpuErrchk(cudaFreeHost(h_partial_sums));
	gpuErrchk(cudaFreeHost(h_hist));
	gpuErrchk(cudaFree(d_data));
	gpuErrchk(cudaFree(d_hist));
	gpuErrchk(cudaFree(d_partial_sums));

	return 0;
}