#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "diffusion.h"
#include <stdio.h>

#ifndef BLOCK_SIZE_3D
#define BLOCK_SIZE_3D 8 
#endif

#ifndef BLOCK_SIZE_1D
#define BLOCK_SIZE_1D 256
#endif

#ifndef JACOBI_ITERATIONS
#define JACOBI_ITERATIONS 40
#endif

#ifndef VISCOSITY
#define VISCOSITY 0.0000148 //air at 15C
#endif

void diffusion(float *&vec_field, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    float *x0, *x1;
    float alpha = (LATTICE_SIZE*LATTICE_SIZE)/(VISCOSITY*DELTA_T);
    float beta = 6 + alpha;

    cudaError cuda_ret = cudaMalloc((void**) &x0, 3*size_x*size_y*size_z*sizeof(float));
    if(cuda_ret != cudaSuccess){
        printf("Error: failed to allocate \"x0\" in func \"diffusion\"\n\tThrew: %s\n", cudaGetErrorString(cuda_ret));fflush(stdout);
	return;
    }
    cuda_ret = cudaMalloc((void**) &x1, 3*size_x*size_y*size_z*sizeof(float));
    if(cuda_ret != cudaSuccess){
	printf("Error: failed to allocate \"x1\" in func \"diffusion\"\n\tThrew: %s\n", cudaGetErrorString(cuda_ret));fflush(stdout);
	return;
    }
    cudaDeviceSynchronize();

    dim3 dimBlock3D(BLOCK_SIZE_3D,BLOCK_SIZE_3D,BLOCK_SIZE_3D);
    dim3 dimGrid3D((size_x-1)/BLOCK_SIZE_3D+1,(size_y-1)/BLOCK_SIZE_3D+1,(size_z-1)/BLOCK_SIZE_3D+1);

    jacobiIteration<<<dimGrid3D, dimBlock3D>>>(vec_field, x0, vec_field, alpha, beta, size_x, size_y, size_z);
    cudaDeviceSynchronize();
    applyVelocityBoundary<<<dimGrid3D, dimBlock3D>>>(x0, size_x, size_y, size_z);
    cudaDeviceSynchronize();

    for(int i=0;i<JACOBI_ITERATIONS;i++) {
        jacobiIteration<<<dimGrid3D, dimBlock3D>>>(x0, x1, vec_field, alpha, beta, size_x, size_y, size_z);
        cudaDeviceSynchronize();
        applyVelocityBoundary<<<dimGrid3D, dimBlock3D>>>(x1, size_x, size_y, size_z);
        cudaDeviceSynchronize();
        float *temp = x0;
        x0 = x1;
        x1 = temp;
    }

    float *temp = vec_field;
    vec_field = x0;
    x0 = temp;

    //vectorCopy<<<1000,(size_x*size_y*size_z*3-1)/1000+1>>>(x0, vec_field, size_x*size_y*size_z*3); 

    cudaFree(x0);
    cudaFree(x1);
    cudaDeviceSynchronize();
}
