#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_


#define BLOCK_SIZE 32
#define NUM_ELEMENTS 512
__global__ void reduction(float *g_data, float *result, int n);


#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
