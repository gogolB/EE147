#include "GPU_Setup.h"
#include <stdio.h>
#include <thrust/host_vector.h>

void initOGL_CUDA()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaGLSetGLDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}