#ifndef TIMING_H
#define TIMING_H

#include "gpu_error_check.h"
cudaEvent_t TIMING_START_EVENT, TIMING_STOP_EVENT;
float TIMING_ELAPSED_TIME;

void TIMING_START()
{
	gpuErrchk(cudaEventCreate(&TIMING_START_EVENT));
	gpuErrchk(cudaEventCreate(&TIMING_STOP_EVENT));
	gpuErrchk(cudaEventRecord(TIMING_START_EVENT, 0));
}

void TIMING_STOP()
{
	gpuErrchk(cudaEventRecord(TIMING_STOP_EVENT, 0));
	gpuErrchk(cudaEventSynchronize(TIMING_STOP_EVENT));
	gpuErrchk(cudaEventElapsedTime(&TIMING_ELAPSED_TIME, TIMING_START_EVENT,
			     TIMING_STOP_EVENT));
	gpuErrchk(cudaEventDestroy(TIMING_START_EVENT));
	gpuErrchk(cudaEventDestroy(TIMING_STOP_EVENT));
}

void TIMING_PRINT()
{
	printf("******** Total Running Time of All Kernels = %f ms *******\n", TIMING_ELAPSED_TIME);
}

#endif