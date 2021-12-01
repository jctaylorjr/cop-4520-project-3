__global__ void histogram(int *d_data, int *d_hist, int n_partitions, int n_elements)
{
	//creating shared memory so that every thread in block can combine values
	extern __shared__ int temp[];

	//initializing BUCKET_TYPE val in temp to 0
	if(threadIdx.x) { 
		temp[threadIdx.x] = 0;
	}
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int start = i;
	int stride = blockDim.x * gridDim.x;

	//uses grid stride and load balancing
	while (i < n_elements) {
		int h = bfe(d_data[i], 31 - (uint)log2f(n_partitions),
			    (uint)log2f(n_partitions));
		/*
		printf("Thread ID: %d\th: %d\td_data[i]: %d\n", i, h,
		       d_data[i]);
		printf("threadIdx.x:%d\tblockIdx.x:%d\tblockDim.x:%d\n",
		       threadIdx.x, blockIdx.x, blockDim.x);
		*/
		atomicAdd(&temp[h], 1);
		i += stride;
	}
	__syncthreads();

	while (start < n_partitions) {
		atomicAdd(&d_hist[threadIdx.x], temp[threadIdx.x]);
		//printf("temp %d: %d\n", threadIdx.x, temp[threadIdx.x]);
		//printf("d_hist %d: %d\n", threadIdx.x, d_hist[threadIdx.x]);
		start += stride;
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
__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)
{
	extern __shared__ int sums[];
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = id % warpSize;
	// determine a warp_id within a block
	int warp_id = threadIdx.x / warpSize;

	// Below is the basic structure of using a shfl instruction
	// for a scan.
	// Record "value" as a variable - we accumulate it along the way
	int value = data[id];

	// Now accumulate in log steps up the chain
	// compute sums, with another thread's value who is
	// distance delta away (i).  Note
	// those threads where the thread 'i' away would have
	// been out of bounds of the warp are unaffected.  This
	// creates the scan sum.

#pragma unroll
	for (int i = 1; i <= width; i *= 2) {
		unsigned int mask = 0xffffffff;
		int n = __shfl_up_sync(mask, value, i, width);

		if (lane_id >= i)
			value += n;
	}

	// value now holds the scan value for the individual thread
	// next sum the largest values for each warp

	// write the sum of the warp to smem
	if (threadIdx.x % warpSize == warpSize - 1) {
		sums[warp_id] = value;
	}

	__syncthreads();

	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
		int warp_sum = sums[lane_id];

		int mask = (1 << (blockDim.x / warpSize)) - 1;
		for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
			int n = __shfl_up_sync(mask, warp_sum, i,
					       (blockDim.x / warpSize));

			if (lane_id >= i)
				warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// perform a uniform add across warps in the block
	// read neighbouring warp's sum and add it to threads value
	int blockSum = 0;

	if (warp_id > 0) {
		blockSum = sums[warp_id - 1];
	}

	value += blockSum;

	// Now write out our result
	data[id] = value;

	// last thread has sum, write write out the block's sum
	if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
		partial_sums[blockIdx.x] = value;
	}
}


//int n_elements = atoi(argv[1]);

//int* elements_array;

//cudaMallocHost((void**)&elements_array, sizeof(int)*n_elements); //use pinned memory

//dataGenerator(elements_array, n_elements, 0, 1);

/* your code */

//cudaFreeHost(s_h);

//return 0;

/*
    FILE *fptr = fopen("sample.txt", "w");
    if (fptr == NULL)
    {
        printf("Could not open file");
        return 0;
    }
   
    for (int i = 0; i < n_elements; i++)
    {
        fprintf(fptr,"%d\n", elements_array[i]);
    }
    fclose(fptr);
	*/

	// int i = 0;
	// while (i < n_partitions) {
	// 	printf("partition %d:\toffset: %d\tnumber of keys: %d.\n", i,
	// 	       h_psums[i], h_hist[i]);
	// 	i++;
	// }

	// int i = 0;
	// while (i < n_elements) {
	// 	printf("Number: %d\n", h_ordered[i]);
	// 	i++;
	// }
