#include <assert.h>

#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))

//data generator
void dataGenerator(int *data, int count, int first, int step)
{
	assert(data != NULL);

	for (int i = 0; i < count; ++i)
		data[i] = first + i * step;
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

//define the histogram kernel here
__global__ void histogram()
{
}

//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
__global__ void prefixScan()
{
}

//define the reorder kernel here
__global__ void Reorder()
{
}

int main(int argc, char const *argv[])
{
	int rSize = atoi(argv[1]);

	int *r_h;

	cudaMallocHost((void **)&r_h, sizeof(int) * rSize); //use pinned memory

	dataGenerator(r_h, rSize, 0, 1);

	/* your code */

	//cudaFreeHost(s_h);

	return 0;
}