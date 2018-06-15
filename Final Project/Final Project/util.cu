#include "util.h"

#ifndef JACOBI_ITERATIONS
#define JACOBI_ITERATIONS 40
#endif

__global__ void zeroVector(float *vector, unsigned int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<n)
        vector[i] = 0;
}

__global__ void jacobiIteration(float *x0, float *x1, float *b, float alpha, float beta,  unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    int bx=blockIdx.x,by=blockIdx.y,bz=blockIdx.z;
    int tx=threadIdx.x,ty=threadIdx.y,tz=threadIdx.z;

    int x=blockDim.x*bx+tx;
    int y=blockDim.y*by+ty;
    int z=blockDim.z*bz+tz;

    if(x<size_x&&y<size_y&&z<size_z) {
        x1[((z*size_y+y)*size_x+x)*3+0] = 0;
        x1[((z*size_y+y)*size_x+x)*3+1] = 0;
        x1[((z*size_y+y)*size_x+x)*3+2] = 0;

        if(x>0&&x<size_x-1) {
            x1[((z*size_y+y)*size_x+x)*3+0] += x0[((z*size_y+y)*size_x+x+1)*3+0]+x0[((z*size_y+y)*size_x+x-1)*3+0];
            x1[((z*size_y+y)*size_x+x)*3+1] += x0[((z*size_y+y)*size_x+x+1)*3+1]+x0[((z*size_y+y)*size_x+x-1)*3+1];
            x1[((z*size_y+y)*size_x+x)*3+2] += x0[((z*size_y+y)*size_x+x+1)*3+2]+x0[((z*size_y+y)*size_x+x-1)*3+2];
        }
        if(y>0&&y<size_y-1) {
            x1[((z*size_y+y)*size_x+x)*3+0] += x0[((z*size_y+y+1)*size_x+x)*3+0]+x0[((z*size_y+y-1)*size_x+x)*3+0];
            x1[((z*size_y+y)*size_x+x)*3+1] += x0[((z*size_y+y+1)*size_x+x)*3+1]+x0[((z*size_y+y-1)*size_x+x)*3+1];
            x1[((z*size_y+y)*size_x+x)*3+2] += x0[((z*size_y+y+1)*size_x+x)*3+2]+x0[((z*size_y+y-1)*size_x+x)*3+2];
        }
        if(z>0&&z<size_z-1) {
            x1[((z*size_y+y)*size_x+x)*3+0] += x0[(((z+1)*size_y+y)*size_x+x)*3+0]+x0[(((z+1)*size_y+y)*size_x+x)*3+0];
            x1[((z*size_y+y)*size_x+x)*3+1] += x0[(((z+1)*size_y+y)*size_x+x)*3+1]+x0[(((z+1)*size_y+y)*size_x+x)*3+1];
            x1[((z*size_y+y)*size_x+x)*3+2] += x0[(((z+1)*size_y+y)*size_x+x)*3+2]+x0[(((z+1)*size_y+y)*size_x+x)*3+2];
        }
        x1[((z*size_y+y)*size_x+x)*3+0] += alpha*b[((z*size_y+y)*size_x+x)];
        x1[((z*size_y+y)*size_x+x)*3+1] += alpha*b[((z*size_y+y)*size_x+x)];
        x1[((z*size_y+y)*size_x+x)*3+2] += alpha*b[((z*size_y+y)*size_x+x)];

        x1[((z*size_y+y)*size_x+x)*3+0]/=beta;
        x1[((z*size_y+y)*size_x+x)*3+1]/=beta;
        x1[((z*size_y+y)*size_x+x)*3+2]/=beta;
    }
}

__global__ void applyPressureBoundary(float *pressure, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    int bx=blockIdx.x,by=blockIdx.y,bz=blockIdx.z;
    int tx=threadIdx.x,ty=threadIdx.y,tz=threadIdx.z;

    int x=blockDim.x*bx+tx;
    int y=blockDim.y*by+ty;
    int z=blockDim.z*bz+tz;

    if(x<size_x&&y<size_y&&z<size_z) {
        if(x==0) {
            pressure[((z*size_y+y)*size_x+x)*3+0] = -1*pressure[((z*size_y+y)*size_x+x+1)*3+0];
        } else if(x==size_x-1) {
            pressure[((z*size_y+y)*size_x+x)*3+0] = -1*pressure[((z*size_y+y)*size_x+x-1)*3+0];
        }
        if(y==0) {
            pressure[((z*size_y+y)*size_x+x)*3+1] = -1*pressure[((z*size_y+y+1)*size_x+x)*3+1];
        } else if(y==size_y-1) {
            pressure[((z*size_y+y)*size_x+x)*3+1] = -1*pressure[((z*size_y+y-1)*size_x+x)*3+1];
        }
        if(z==0) {
            pressure[((z*size_y+y)*size_x+x)*3+2] = -1*pressure[(((z+1)*size_y+y)*size_x+x)*3+2];
        } else if(z==size_z-1) {
            pressure[((z*size_y+y)*size_x+x)*3+2] = -1*pressure[(((z-1)*size_y+y)*size_x+x)*3+2];
        }
    }
}

__global__ void applyVelocityBoundary(float *velocity, unsigned int size_x, unsigned int size_y, unsigned int size_z) {
    int bx=blockIdx.x, by=blockIdx.y, bz=blockIdx.z;
    int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;

    int x=bx*blockDim.x+tx;
    int y=by*blockDim.y+ty;
    int z=bz*blockDim.z+tz;

    if(x<size_x && y<size_y && z<size_z) {
        if(x==0) {
            velocity[((z*size_y+y)*size_x+x)*3+0] = -1*velocity[((z*size_y+y)*size_x+x+1)*3+0];
            velocity[((z*size_y+y)*size_x+x)*3+1] = -1*velocity[((z*size_y+y)*size_x+x+1)*3+1];
            velocity[((z*size_y+y)*size_x+x)*3+2] = -1*velocity[((z*size_y+y)*size_x+x+1)*3+2];
        } else if(x+1==size_x) {
            velocity[((z*size_y+y)*size_x+x)*3+0] = -1*velocity[((z*size_y+y)*size_x+x-1)*3+0];
            velocity[((z*size_y+y)*size_x+x)*3+1] = -1*velocity[((z*size_y+y)*size_x+x-1)*3+1];
            velocity[((z*size_y+y)*size_x+x)*3+2] = -1*velocity[((z*size_y+y)*size_x+x-1)*3+2];
        } else if(y==0) {
            velocity[((z*size_y+y)*size_x+x)*3+0] = -1*velocity[((z*size_y+y+1)*size_x+x)*3+0];
            velocity[((z*size_y+y)*size_x+x)*3+1] = -1*velocity[((z*size_y+y+1)*size_x+x)*3+1];
            velocity[((z*size_y+y)*size_x+x)*3+2] = -1*velocity[((z*size_y+y+1)*size_x+x)*3+2];
        } else if(y+1==size_y) {
            velocity[((z*size_y+y)*size_x+x)*3+0] = -1*velocity[((z*size_y+y-1)*size_x+x)*3+0];
            velocity[((z*size_y+y)*size_x+x)*3+1] = -1*velocity[((z*size_y+y-1)*size_x+x)*3+1];
            velocity[((z*size_y+y)*size_x+x)*3+2] = -1*velocity[((z*size_y+y-1)*size_x+x)*3+2];
        } else if(z==0) {
            velocity[(((z)*size_y+y)*size_x+x)*3+0] = -1*velocity[(((z+1)*size_y+y)*size_x+x)*3+0];
            velocity[(((z)*size_y+y)*size_x+x)*3+1] = -1*velocity[(((z+1)*size_y+y)*size_x+x)*3+1];
            velocity[(((z)*size_y+y)*size_x+x)*3+2] = -1*velocity[(((z+1)*size_y+y)*size_x+x)*3+2];
        } else if(z+1==size_z) {
            velocity[(((z)*size_y+y)*size_x+x)*3+0] = -1*velocity[(((z-1)*size_y+y)*size_x+x)*3+0];
            velocity[(((z)*size_y+y)*size_x+x)*3+1] = -1*velocity[(((z-1)*size_y+y)*size_x+x)*3+1];
            velocity[(((z)*size_y+y)*size_x+x)*3+2] = -1*velocity[(((z-1)*size_y+y)*size_x+x)*3+2];
        }
    }
}

__global__ void vectorCopy(float *src, float *dst, unsigned int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    if(i<n) {
        dst[i] = src[i];
    }
}

/*__device__ unsigned int roundUp(float val) {
    return (unsigned int)(val/LATTICE_SIZE)+1;
}

__device__ unsigned int roundDown(float val) {
    return (unsigned int)(val/LATTICE_SIZE);
}*/

