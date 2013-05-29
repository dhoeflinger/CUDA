

// includes, kernels
#include <assert.h>


#include "scan_largearray_kernel.h"

// MP4.2 - Host Helper Functions (allocate your own data structure...)

float ** BlockSums;

void preallocBlockSums(int num_elements)
{
	int n = num_elements;
	int i = 0;


	while ((n = ceil((float)n /  (float)(BLOCK_SIZE*2))) > 1)
	{
		i ++;
	}
	i++;

	BlockSums = (float**)malloc(sizeof(float*) * i);
	n = num_elements;
	i = 0;
	while ((n = ceil((float)n /  (float)(BLOCK_SIZE*2))) > 1)
	{
		cudaMalloc((void**)&(BlockSums[i]), n * sizeof(float));
		i++;
	}
	cudaMalloc((void**)&(BlockSums[i]), n * sizeof(float));

}

// MP4.2 - Device Functions


// MP4.2 - Kernel Functions


__global__ void AdjustIncr (float * arr, float * incr, int n)
{
	if(blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x + 1 < n)
	{
		arr[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x + 1] += incr[blockIdx.x];
		arr[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x] += incr[blockIdx.x];
	}
	else if (blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x < n)
	{
		arr[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x] += incr[blockIdx.x];
	}
}


__global__ void prescanArrayKernel(float *outArray, float * inArray, int numElements, float *blockSums)
{

	__shared__ float temp[BLOCK_SIZE * 2];
	int tid= threadIdx.x;
	int offset = 1;

	int start = (BLOCK_SIZE * 2) * blockIdx.x;

	if(numElements > start + 2 * tid)
	{
		temp[2 * tid ] = inArray[start + 2*tid];
		if(numElements > start + 2 * tid + 1)
		{
			temp[2 * tid + 1] = inArray[start + 2*tid + 1];
		}
		else
		{
			temp[2 * tid + 1] = 0;
		}
	}
	else 
	{
		temp[2 * tid ] = 0;
		temp[2 * tid + 1] = 0;
	}


	for (int d = BLOCK_SIZE; d>0; d>>=1)
	{
		__syncthreads();
		if(tid < d)
		{
			int ai = offset * (2 * tid + 1) -1;
			int bi = offset * (2 * tid + 2) -1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (tid == 0) 
	{
		temp[BLOCK_SIZE * 2 - 1] = 0;
	}

	for(int d = 1; d < BLOCK_SIZE * 2; d*=2)
	{
		offset >>= 1;
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) -1;
			int bi = offset * (2 * tid + 2) -1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
 		}
	}
	__syncthreads();


	if(numElements > start + 2 * tid + 1)
	{
		outArray[start + 2*tid + 1 ] = temp[2 * tid + 1];
		outArray[start + 2*tid] = temp[2 * tid];
	}
	else 
	{
		if(numElements > start + 2 * tid)
		{
			outArray[start + 2*tid] = temp[2 * tid];
		}
	}

	if(tid == 0)
		blockSums[blockIdx.x] = temp[2 * BLOCK_SIZE - 1];

}

// **===-------- MP4.2 - Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

void prescanArrayHelper(float *outArray, float *inArray, int numElements, int index)
{
	dim3 dim_block, dim_grid;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = dim_block.z = 1;

	dim_grid.x = ceil((float)(numElements / (float)(dim_block.x * 2)));
	dim_grid.y = dim_grid.z = 1;

	prescanArrayKernel<<<dim_grid,dim_block>>>(outArray, inArray, numElements, BlockSums[index]);

	if (dim_grid.x > 1)
	{
		prescanArrayHelper(BlockSums[index], BlockSums[index], dim_grid.x, index+1);
		AdjustIncr<<<dim_block, dim_grid>>>(outArray, BlockSums[index], numElements);
	}

}

void prescanArray(float *outArray, float *inArray, int numElements)
{
	prescanArrayHelper(outArray,inArray, numElements, 0);
}
// **===-----------------------------------------------------------===**


