#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "points.h"
#include <stdio.h>


__device__ unsigned int roundUp2(float val) {
    return (unsigned int)(val/LATTICE_SIZE)+1;
}

__device__ unsigned int roundDown2(float val) {
    return (unsigned int)(val/LATTICE_SIZE);
}

__device__ bool checkBound2(unsigned int x0,unsigned int x1,unsigned int y0,unsigned int y1,unsigned int z0,unsigned int z1,unsigned int size_x,unsigned int size_y,unsigned int size_z) {
    return (x0<size_x&&x1<size_x)&&
           (y0<size_y&&y1<size_y)&&
           (z0<size_z&&z1<size_z);
}

__global__ void calculateUpdate(float *point_vec, float *velocity_field, float *point_velocity, unsigned int n, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;

    if(i<n) {
        unsigned int x_0=roundDown2(point_vec[7*i]), x_1=roundUp2(point_vec[7*i]);
        unsigned int y_0=roundDown2(point_vec[7*i+1]), y_1=roundUp2(point_vec[7*i+1]);
        unsigned int z_0=roundDown2(point_vec[7*i+2]), z_1=roundUp2(point_vec[7*i+2]);
	if(checkBound2(x_0,x_1,y_0,y_1,z_0,z_1,size_x,size_y,size_z)) {
            float x_d = (point_vec[7*i]-x_0*LATTICE_SIZE)/(x_1-point_vec[7*i] * LATTICE_SIZE);
            float y_d = (point_vec[7*i+1]-y_0*LATTICE_SIZE)/(y_1-point_vec[7*i+1] * LATTICE_SIZE);
            float z_d = (point_vec[7*i+2]-z_0*LATTICE_SIZE)/(z_1-point_vec[7*i+2] * LATTICE_SIZE);

            float c8[8];
            float c4[4];
            float c2[2];
            for(int l=0;l<3;l++) {
                c8[0] = velocity_field[((z_0*size_y+y_0)*size_x+x_1)*3+l];
                c8[1] = velocity_field[((z_1*size_y+y_0)*size_x+x_1)*3+l];
                c8[2] = velocity_field[((z_0*size_y+y_0)*size_x+x_0)*3+l];
                c8[3] = velocity_field[((z_1*size_y+y_0)*size_x+x_0)*3+l];
                c8[4] = velocity_field[((z_0*size_y+y_1)*size_x+x_1)*3+l];
                c8[5] = velocity_field[((z_1*size_y+y_1)*size_x+x_1)*3+l];
                c8[6] = velocity_field[((z_0*size_y+y_1)*size_x+x_0)*3+l];
                c8[7] = velocity_field[((z_1*size_y+y_1)*size_x+x_0)*3+l];

                for(int j=0;j<4;j++)
                    c4[j] = c8[j]*(1-x_d)+c8[j+4]*x_d;

                for(int j=0;j<2;j++)
                    c2[j] = c4[j]*(1-y_d)+c4[j+2]*y_d;

                point_velocity[7*i+l] = /*c2[0]*(1-z_d)+c2[1]*z_d+*/point_vec[7*i+l+3];
            } 
	} else {
		if (i == 0) {
			printf("%d, %d, %d, %d, %d, %d\n", x_0, x_1, y_0, y_1, z_0, z_1);
		}
        point_velocity[7*i] = 0;
	    point_velocity[7*i+1] = 0;
	    point_velocity[7*i+2] = 0;
	}
	
    point_velocity[7*i+3] = 0;
 	point_velocity[7*i+4] = 0;
	point_velocity[7*i+5] = -9.8;
	point_velocity[7*i+6] = -1;
    }
	if (i == 0) {
		printf("[%f,%f,%f,%f,%f,%f,%f]\n", point_velocity[0], point_velocity[1], point_velocity[2], point_velocity[3], point_velocity[4], point_velocity[5], point_velocity[6]);
	}
}

__global__ void increment(float *points, float *velocity, unsigned int n) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;

    if(i<n) {
        points[i] += velocity[i]*DELTA_T;
    }
}

void expirePoints(float *points, unsigned int n, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    for(int i=0;i<n;i++)
        if(points[i*7+6]<=0||points[i*7]<=0||points[i*7]>=size_x*LATTICE_SIZE||points[i*7+1]<=0||points[i*7+1]>=size_y*LATTICE_SIZE||points[i*7+2]<=0||points[i*7+2]>=size_z*LATTICE_SIZE) { //expire point it T<=0 or it is outside bounds
            points[i*7] = 1 + (-5 + rand()%10)/100.0;
	        points[i*7+1] = 1 + (-5 + rand()%10)/100.0;
	        points[i*7+2] = 1 + (-5 + rand()%10)/100.0;
            points[i*7+3] = (-192 + rand()%384)/10.0;
	        points[i*7+4] = (-149 + rand()%298)/10.0;
	        points[i*7+5] = (-192 + rand()%384)/10.0;
	        points[i*7+6] = 10;
	}
}

/*__global__ void randInit(curandState_t *states, unsigned int seed, unsigned int n) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;

    if(i<n) {
        curand_init(seed, i, 0, &states[i]);
    }
}*/

void updatePoints(float *velocity_field,  float *points, unsigned int n, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    float  *velocity;
    cudaError cuda_ret;
    cuda_ret = cudaMalloc((void **) &velocity, 7*n*sizeof(float));
    if(cuda_ret != cudaSuccess) {
        printf("Error failed to allocate memory in func \"updatePoints\"\n\tThrew error: %s\n", cudaGetErrorString(cuda_ret));fflush(stdout);
	return;
    }
    cudaDeviceSynchronize();

    dim3 DimBlock(1000,1,1);
    dim3 DimGrid((n-1)/1000+1,1,1);

    calculateUpdate<<<DimGrid, DimBlock>>>(points, velocity_field, velocity, n, size_x, size_y, size_z);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) {
        printf("Error failed to launch kernel \"calculateUpdate\" in func \"updatePoints\"\n");fflush(stdout);
        return;
    }

    dim3 DimBlock_1D(1000,1,1);
    dim3 DimGrid_1D((n*7-1)/1000+1,1,1);

    increment<<<DimGrid_1D, DimBlock_1D>>>(points, velocity, n);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) {
        printf("Error failed to launch kernel \"increment\" in func \"updatePoints\"\n");fflush(stdout);
        return;
    }

    expirePoints(points, n, size_x, size_y, size_z);

    cudaFree(velocity);
}

void initPoints(float *points, unsigned int n, unsigned int size_x, unsigned int size_y, unsigned int size_z){
    srand(217);

    dim3 DimBlock(1000,1,1);
    dim3 DimGrid((7*n-1)/1000+1,1,1);

    dim3 DimBlock_2(1000,1,1);
    dim3 DimGrid_2((n-1)/1000+1,1,1);

    zeroVector<<<DimGrid, DimBlock>>>(points, n*7);
    cudaDeviceSynchronize();

    expirePoints(points, n, size_x, size_y, size_z);
}
