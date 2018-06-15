#ifndef __DIFFUSION_H__
#define __DIFFUSION_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "../util.h"
#ifndef LATTICE_SIZE
#define LATTICE_SIZE 0.01
#endif

#ifndef DELTA_T
#define DELTA_T 0.0167
#endif


void diffusion(float *&vec_field, unsigned int size_x, unsigned int size_y, unsigned int size_z);

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
