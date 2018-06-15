#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "advection.h"

#ifndef BLOCK_SIZE_3D
#define BLOCK_SIZE_3D 8
#endif

#ifndef BLOCK_SIZE_1D
#define BLOCK_SIZE_1D 256
#endif

__device__ unsigned int roundUp(float val) {
    return (unsigned int)(val/LATTICE_SIZE)+1;
}

__device__ unsigned int roundDown(float val) {
    return (unsigned int)(val/LATTICE_SIZE);
}

__device__ bool checkBound(unsigned int x0,unsigned int x1,unsigned int y0,unsigned int y1,unsigned int z0,unsigned int z1,unsigned int size_x,unsigned int size_y,unsigned int size_z) {
    return (x0<size_x&&x1<size_x)&&
	   (y0<size_y&&y1<size_y)&&
	   (z0<size_z&&z1<size_z);
}

__global__ void interpolate(float *pos, float *field, float *val, unsigned int n, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    if(i<n) {
        unsigned int x_0=roundDown(pos[3*i]), x_1=roundUp(pos[3*i]);
        unsigned int y_0=roundDown(pos[3*i+1]), y_1=roundUp(pos[3*i+1]);
        unsigned int z_0=roundDown(pos[3*i+2]), z_1=roundUp(pos[3*i+2]);
        if(checkBound(x_0,x_1,y_0,y_1,z_0,z_1,size_x,size_y,size_z)) {
            float x_d = (pos[3*i]-x_0*LATTICE_SIZE)/(x_1*LATTICE_SIZE -pos[3*i]);
            float y_d = (pos[3*i+1]-y_0*LATTICE_SIZE)/(y_1*LATTICE_SIZE -pos[3*i+1]);
            float z_d = (pos[3*i+2]-z_0*LATTICE_SIZE)/(z_1*LATTICE_SIZE -pos[3*i+2]);
			
	    float c8[8];
	    float c4[4];
	    float c2[2];
	    for(int l=0;l<3;l++) {
                c8[0] = field[((z_0*size_y+y_0)*size_x+x_1)*3+l];
                c8[1] = field[((z_1*size_y+y_0)*size_x+x_1)*3+l];
                c8[2] = field[((z_0*size_y+y_0)*size_x+x_0)*3+l];
                c8[3] = field[((z_1*size_y+y_0)*size_x+x_0)*3+l];
                c8[4] = field[((z_0*size_y+y_1)*size_x+x_1)*3+l];
                c8[5] = field[((z_1*size_y+y_1)*size_x+x_1)*3+l];
                c8[6] = field[((z_0*size_y+y_1)*size_x+x_0)*3+l];
                c8[7] = field[((z_1*size_y+y_1)*size_x+x_0)*3+l];

                for(int j=0;j<4;j++)
                    c4[j] = c8[j]*(1-x_d)+c8[j+4]*x_d;

                for(int j=0;j<2;j++)
                    c2[j] = c4[j]*(1-y_d)+c4[j+2]*y_d;

	        val[3*i+l] = c2[0]*(1-z_d)+c2[1]*z_d;
	    }
	} else {
            val[3*i]=0;
	    val[3*i+1]=0;
	    val[3*i+2]=0;
	}
    }
}

__global__ void reverseStep(float *field, float *pos, unsigned int n, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;

    unsigned int z = i/(size_x*size_y);
    unsigned int y = (i-z*(size_x*size_y))/size_x;
    unsigned int x = i-(z*size_y+y)*size_x;

    if(i<n) {
        pos[3*i] = x*LATTICE_SIZE - field[((z*size_y+y)*size_x+x)*3];
	pos[3*i+1] = x*LATTICE_SIZE - field[((z*size_y+y)*size_x+x)*3+1];
	pos[3*i+2] = x*LATTICE_SIZE - field[((z*size_y+y)*size_x+x)*3+2];
    }
}

void advection(float *&vec_field, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    unsigned int n = size_x*size_y*size_z;

    float *pos, *temp;
    cudaMalloc((void**) &pos, n*3*sizeof(float));
    cudaMalloc((void**) &temp, n*3*sizeof(float));
    cudaDeviceSynchronize();
    dim3 dimBlock(BLOCK_SIZE_1D,1,1);
    dim3 dimGrid((n-1)/BLOCK_SIZE_1D+1,1,1);

    reverseStep<<<dimGrid, dimBlock>>>(vec_field, pos, n, size_x, size_y, size_z);
    cudaDeviceSynchronize();
    interpolate<<<dimGrid, dimBlock>>>(pos, vec_field, temp, n, size_x, size_y, size_z);
    cudaDeviceSynchronize();
    applyVelocityBoundary<<<dimGrid, dimBlock>>>(temp, size_x, size_y, size_z);
    float *temp_swap = vec_field;
    vec_field = temp;
    temp = temp_swap;
    //vectorCopy<<<1000,(n*3-1)/1000+1>>>(temp, vec_field, n*3);

    cudaError cuda_ret = cudaFree(pos);
    if(cuda_ret != cudaSuccess)
        printf("%s\n",cudaGetErrorString(cuda_ret)); fflush(stdout);
    cuda_ret = cudaFree(temp);
    if(cuda_ret != cudaSuccess)
	printf("%s\n",cudaGetErrorString(cuda_ret)); fflush(stdout);
    cudaDeviceSynchronize();
}
