#ifndef __UTIL_H__
#define __UTIL_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LATTICE_SIZE
#define LATTICE_SIZE 0.01
#endif

#ifndef DELTA_T
#define DELTA_T 0.0167
#endif


__global__ void zeroVector(float *vec, unsigned int n);
__global__ void jacobiIteration(float *x0, float *x1, float *b, float alpha, float beta,  unsigned int size_x, unsigned int size_y, unsigned int size_z);
__global__ void applyPressureBoundary(float *pressure, unsigned int size_x, unsigned int size_y, unsigned int size_z);
__global__ void applyVelocityBoundary(float *velocity, unsigned int size_x, unsigned int size_y, unsigned int size_z);
__global__ void vectorCopy(float *src, float *dst, unsigned int n);
//__device__ unsigned int roundUp(float);
//__device__ unsigned int roundDown(float);
#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
	    do {\
		            fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
		            exit(-1);\
		        } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
