#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "force.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8 
#endif

__global__ void applyLocalForce(float *initial_field, float *final_field, float *force, float *pos, float r, unsigned int size_x, unsigned int size_y, unsigned int size_z) { 
    int bx=blockIdx.x, by=blockIdx.y, bz=blockIdx.z;
    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;

    int x=bx*blockDim.x+tx;
    int y=by*blockDim.y+ty;
    int z=bz*blockDim.z+tz;

    if(x<size_x && y<size_y && z<size_z) {
        if(sqrt(pow(pos[0] - x*LATTICE_SIZE,2) + pow(pos[1] - y*LATTICE_SIZE,2) + pow(pos[2] - z*LATTICE_SIZE,2)) < r) {
            float gaussian = exp(-1*(pow(pos[0]-x*LATTICE_SIZE,2)+pow(pos[1]-y*LATTICE_SIZE,2)+pow(pos[2]-z*LATTICE_SIZE,2))/r);
            final_field[((z*size_y+y)*size_x+x)*3+0] = initial_field[((z*size_y+y)*size_x+x)*3+0] + force[0]*DELTA_T*gaussian;
	    final_field[((z*size_y+y)*size_x+x)*3+1] = initial_field[((z*size_y+y)*size_x+x)*3+1] + force[1]*DELTA_T*gaussian; 
	    final_field[((z*size_y+y)*size_x+x)*3+2] = initial_field[((z*size_y+y)*size_x+x)*3+2] + force[2]*DELTA_T*gaussian;
	} else {
            final_field[((z*size_y+y)*size_x+x)*3+0] = initial_field[((z*size_y+y)*size_x+x)*3+0];
	    final_field[((z*size_y+y)*size_x+x)*3+1] = initial_field[((z*size_y+y)*size_x+x)*3+1];
	    final_field[((z*size_y+y)*size_x+x)*3+2] = initial_field[((z*size_y+y)*size_x+x)*3+2];
	}
    }
}

__global__ void applyGlobalForce(float *initial_field, float *final_field, float *force, unsigned int n) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;

    if(i<n) {
        final_field[i] = initial_field[i] + force[i%3]*DELTA_T;
    }

}

void localForce(float *&vec_field, float *force, float *pos, float r, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    dim3 DimBlock(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
    dim3 DimGrid((size_x-1)/BLOCK_SIZE+1,(size_y-1)/BLOCK_SIZE+1,(size_z-1)/BLOCK_SIZE+1);

    float *v2, *temp;
    cudaMalloc((void**) &v2, 3*size_x*size_y*size_z*sizeof(float));

    applyLocalForce<<<DimGrid, DimBlock>>>(vec_field, v2, force, pos, r, size_x, size_y, size_z);
    applyVelocityBoundary<<<DimGrid, DimBlock>>>(v2, size_x, size_y, size_z);

    temp = vec_field;
    vec_field = v2;
    v2 = temp;

    cudaFree(v2);
}

void globalForce(float *&vec_field, float *force, unsigned int size_x, unsigned int size_y, unsigned int size_z){
    unsigned int n = 3*size_x*size_y*size_z;

    dim3 DimBlock(1000,1,1);
    dim3 DimGrid((n-1)/1000+1,1,1);

    dim3 DimBlock_3D(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
    dim3 DimGrid_3D((size_x-1)/BLOCK_SIZE+1,(size_y-1)/BLOCK_SIZE+1,(size_z-1)/BLOCK_SIZE+1);

    float *v2, *temp;
    cudaMalloc((void**) &v2,n*sizeof(float));
    cudaDeviceSynchronize();

    applyGlobalForce<<<DimGrid, DimBlock>>>(vec_field, v2, force, n);
    applyVelocityBoundary<<<DimGrid_3D, DimBlock_3D>>>(v2, size_x, size_y, size_z);

    temp = vec_field;
    vec_field = v2;
    v2 = temp;

    cudaFree(v2);
};
