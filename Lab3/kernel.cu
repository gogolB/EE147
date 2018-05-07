/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to
__global__ void gpuHistogram(unsigned int* data, unsigned int* bins, unsigned int numElements, unsigned int numBins)
{
	extern __shared__ unsigned int privateHistogram[];
	
	int i = 0;
	int stride = blockDim.x * gridDim.x;
	// This is because we have less threads then we do bins and we need to clear all the bins.
	while(i*blockDim.x + threadIdx.x < numBins)
	{
		privateHistogram[i*blockDim.x + threadIdx.x] = 0;
		i++;
	}
	__syncthreads();

	// Normally go through the data.
	i = blockDim.x * blockIdx.x + threadIdx.x;

	while(i < numElements)
	{
		atomicAdd(&(privateHistogram[data[i]]), 1);
		i += stride;
	}

	__syncthreads();
	
	// Copy all the data back.
	i = 0;
	while(i*blockDim.x + threadIdx.x < numBins)
	{
		atomicAdd(&(bins[i*blockDim.x + threadIdx.x]), privateHistogram[i*blockDim.x + threadIdx.x]);
		i++;
	}
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) 
{

    dim3 gridDim(30,1,1);	// 30 SMPs per machine
    dim3 blockDim(32,1,1);	// 32 threads executing at once.

    gpuHistogram<<<gridDim, blockDim, num_bins * sizeof(unsigned int)>>>(input,bins,num_elements,num_bins);

}


