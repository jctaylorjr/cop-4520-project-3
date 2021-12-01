#include <assert.h>
#include <iostream>
#include <limits.h>
#include "timing.cuh"
#include "gpu_error_check.h"

#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))

// data generator
void dataGenerator(int *data, int count, int first, int step)
{
	assert(data != NULL);

	for (int i = 0; i < count; ++i) {
		data[i] = RAND_RANGE(INT_MAX);
	}
	srand(time(NULL));
	for (int i = count - 1; i > 0; i--) // knuth shuffle
	{
		int j = RAND_RANGE(i);
		int k_tmp = data[i];
		data[i] = data[j];
		data[j] = k_tmp;
	}
}

// This function embeds PTX code of CUDA to extract bit field from x.
// "start" is the starting bit position relative to the LSB.
// "nbits" is the bit field length.
// It returns the extracted bit field as an unsigned integer.
__device__ uint bfe(uint x, uint start, uint nbits)
{
	uint bits;
	asm("bfe.u32 %0, %1, %2, %3;"
	    : "=r"(bits)
	    : "r"(x), "r"(start), "r"(nbits));
	return bits;
}

__device__ unsigned int int_to_int(unsigned int k)
{
	return (k == 0 || k == 1 ? k : ((k % 2) + 10 * int_to_int(k / 2)));
}

// define the histogram kernel here
__global__ void histogram(int *d_data, int *d_hist, int n_partitions,
			  int n_elements)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	// uses grid stride
	while (i < n_elements) {
		int h = bfe(d_data[i], 31 - (uint)log2f(n_partitions),
			    (uint)log2f(n_partitions));
		atomicAdd(&d_hist[h], 1);
		i += stride;
	}
}

// define the prefix scan kernel here
// implement it yourself or borrow the code from CUDA samples
__global__ void prefixScan(int *data, int n_partitions, int *d_psums)
{
	// shared mem size of max partitions
	__shared__ int cache[1024];

	// added gride stride variable
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	// this loop is setting all the sums to the
	// previous partitions histogram value
	// block size is messing with the output
	int temp = 0;
	while (tid < n_partitions) {
		cache[tid] = data[tid];
		for (int j = tid - 1; j >= 0; j--) {
			temp += cache[j];
		}
		d_psums[tid] = temp;
		__syncthreads();
		tid += stride;
	}
	__syncthreads();
}

// define the reorder kernel here
__global__ void Reorder(int *d_data, int *d_hist, int n_partitions,
			int n_elements, int *d_psums, int *d_ordered)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	// uses grid stride
	while (i < n_elements) {
		int h = bfe(d_data[i], 31 - (uint)log2f(n_partitions),
			    (uint)log2f(n_partitions));
		offset = atomicAdd(&d_psums[h], 1);
		d_ordered[offset] = d_data[i];
		i += stride;
	}
}

int main(int argc, char *argv[])
{
	int cuda_device = 0;
	cudaDeviceProp deviceProp;
	gpuErrchk(cudaGetDeviceProperties(&deviceProp, cuda_device));

	// default values, and execution arguments are assigned
	int n_elements = 1000000;
	int n_partitions = 1024;

	if (argc > 1) {
		n_elements = atoi(argv[1]);
	}
	if (argc > 2) {
		n_partitions = atoi(argv[2]);
	}

	// setting block and grid size
	int blockSize = 1024;
	int gridSize = (((n_elements / blockSize) + blockSize) / blockSize) *
		       blockSize;

	// variables in host memory
	int *h_data, *h_hist, *h_psums, *h_ordered;
	// variables in device memory
	int *d_data, *d_hist, *d_psums, *d_ordered;

	// allocation space for array used in number generator
	gpuErrchk(cudaMallocHost((void **)&h_data, n_elements * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_data, n_elements * sizeof(int)));

	// allocating arrays used for ordered version of number generator
	gpuErrchk(
		cudaMallocHost((void **)&h_ordered, n_elements * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_ordered, n_elements * sizeof(int)));

	// allocating arrays used to hold histogram values
	gpuErrchk(cudaMallocHost((void **)&h_hist, n_partitions * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_hist, n_partitions * sizeof(int)));

	// allocating arrays used to hold prefix sums
	gpuErrchk(
		cudaMallocHost((void **)&h_psums, n_partitions * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&d_psums, n_partitions * sizeof(int)));

	// using data generator and copying values to device
	dataGenerator(h_data, n_elements, 0, 1);
	gpuErrchk(cudaMemcpy(d_data, h_data, n_elements * sizeof(int),
			     cudaMemcpyHostToDevice));

	// starting timer for kernel executions
	TIMING_START();

	// histogram kernel launch and copying memory back to host
	histogram<<<gridSize, blockSize>>>(d_data, d_hist, n_partitions,
					   n_elements);
	gpuErrchk(cudaMemcpy(h_hist, d_hist, sizeof(int) * n_partitions,
			     cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	// running prefixscan kernel and copying sum values back to host
	prefixScan<<<gridSize, blockSize>>>(d_hist, n_partitions, d_psums);
	gpuErrchk(cudaMemcpy(h_psums, d_psums, sizeof(int) * n_partitions,
			     cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	// launching reorder kernel and coping back to host
	Reorder<<<gridSize, blockSize>>>(d_data, d_hist, n_partitions,
					 n_elements, d_psums, d_ordered);
	gpuErrchk(cudaMemcpy(h_ordered, d_ordered, sizeof(int) * n_elements,
			     cudaMemcpyDeviceToHost));
	gpuErrchk(cudaDeviceSynchronize());

	// stopping timer for kernel executions
	TIMING_STOP();
	TIMING_PRINT();

	// printing out partition, offset, and key values to text file
	FILE *partitions_txt = fopen("PARTITIONS.txt", "w");
	if (partitions_txt == NULL) {
		printf("Could not open file");
		return 0;
	}
	for (int i = 0; i < n_partitions; i++) {
		fprintf(partitions_txt, "partition %d:\toffset: %d\tnumber of keys: %d.\n", i,
			h_psums[i], h_hist[i]);
	}
	fclose(partitions_txt);
	
	// printing out ordered version of numbers from generator,
	// sorted by Reorder kernel
	FILE *reorder_txt = fopen("REORDER.txt", "w");
	if (reorder_txt == NULL) {
		printf("Could not open file");
		return 0;
	}
	for (int i = 0; i < n_elements; i++) {
		fprintf(reorder_txt,"%d\n", h_ordered[i]);
	}
	fclose(reorder_txt);
	
	return 0;
}