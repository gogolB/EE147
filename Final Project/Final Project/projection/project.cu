#include "project.h"

#ifndef BLOCK_SIZE_3D
#define BLOCK_SIZE_3D 8 
#endif

#ifndef BLOCK_SIZE_1D
#define BLOCK_SIZE_1D 256
#endif

#ifndef JACOBI_ITERATIONS
#define JACOBI_ITERATIONS 40
#endif

__global__ void divergence(float *vec_field, float *div, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    int bx=blockIdx.x,by=blockIdx.y,bz=blockIdx.z;
    int tx=threadIdx.x,ty=threadIdx.y,tz=threadIdx.z;

    int x=blockDim.x*bx+tx;
    int y=blockDim.y*by+ty;
    int z=blockDim.z*bz+tz;

    if(x<size_x&&y<size_y&&z<size_z) {
        if(x>0&&(x+1)<size_x&&y>0&&(y+1)<size_y&&z>0&&(z+1)<size_z) {
            float dx = (vec_field[((z*size_y+y)*size_x+x+1)*3+0]-vec_field[((z*size_y+y)*size_x+x-1)*3+0])/(2*DELTA_T);
	    float dy = (vec_field[((z*size_y+y-1)*size_x+x)*3+1]-vec_field[((z*size_y+y-1)*size_x+x)*3+1])/(2*DELTA_T);
	    float dz = (vec_field[(((z+1)*size_y+y)*size_x+x)*3+2]-vec_field[(((z-1)*size_y+y)*size_x+x)*3+2])/(2*DELTA_T);
	    div[(z*size_y+y)*size_x+x] = dx+dy+dz;
	} else {
            div[(z*size_y+y)*size_x+x] = 0;
	}
    }
}

__global__ void vectorSubtract(float *vec_field, float *pressure, float *projected_field, unsigned int n) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;

    if(i<n) {
        projected_field[i] = vec_field[i] - pressure[i];
    }
}

void project(float *&vec_field, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    float *w, *p_k0, *p_k1;
    float alpha = -1*(DELTA_T*DELTA_T);
    float beta = 6;
    
    cudaMalloc((void**) &w, sizeof(float)*size_x*size_y*size_z);
    cudaMalloc((void**) &p_k0, sizeof(float)*size_x*size_y*size_z*3);
    cudaMalloc((void**) &p_k1, sizeof(float)*size_x*size_y*size_z*3);

    cudaDeviceSynchronize();

    dim3 dimBlock3D(BLOCK_SIZE_3D,BLOCK_SIZE_3D,BLOCK_SIZE_3D);
    dim3 dimGrid3D((size_x-1)/BLOCK_SIZE_3D+1,(size_y-1)/BLOCK_SIZE_3D+1,(size_z-1)/BLOCK_SIZE_3D+1);

    dim3 dimBlock1D(BLOCK_SIZE_1D,1,1);
    dim3 dimGrid1D((size_x*size_y*size_z*3-1)/BLOCK_SIZE_1D,1,1);

    divergence<<<dimGrid3D, dimBlock3D>>>(vec_field, w, size_x, size_y, size_z);
    zeroVector<<<dimGrid1D, dimBlock1D>>>(p_k0, size_x*size_y*size_z*3);
    cudaDeviceSynchronize();
    for(int i=0;i<JACOBI_ITERATIONS;i++) {
        jacobiIteration<<<dimGrid3D, dimBlock3D>>>(p_k0, p_k1, w, alpha, beta, size_x, size_y, size_z);
        cudaDeviceSynchronize();
        applyPressureBoundary<<<dimGrid3D, dimBlock3D>>>(p_k1, size_x, size_y, size_z);
        cudaDeviceSynchronize();
        float *temp = p_k0;
        p_k0 = p_k1;
        p_k1 = temp;
    }

    vectorSubtract<<<dimGrid1D, dimBlock1D>>>(vec_field, p_k0, p_k1, size_x*size_y*size_z*3);

    float *temp = vec_field;
    vec_field = p_k1;
    p_k1 = temp;

    cudaFree(w);
    cudaFree(p_k0);
    cudaFree(p_k1);
}
