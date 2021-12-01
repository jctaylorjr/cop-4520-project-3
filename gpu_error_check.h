#ifndef GPU_ERROR_CHECK
#define GPU_ERROR_CHECK

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define gpuErrchk(ans)                                                         \
	{                                                                      \
		gpuAssert((ans), __FILE__, __LINE__);                          \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line,
		      bool abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr,
			ANSI_COLOR_RED "GPUassert: %s %s %d\n" ANSI_COLOR_RESET,
			cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#endif