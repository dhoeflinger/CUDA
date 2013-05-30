

// includes, kernels
#include <assert.h>

#include "helper_cuda.h"
#include "scan_largearray_kernel.h"

// MP4.2 - Host Helper Functions (allocate your own data structure...)

float ** BlockSums;
float ** BlockSumsSummed;
float ** HostSums;
float ** HostSumsSummed;
int * sizes;

#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

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
	BlockSumsSummed = (float**)malloc(sizeof(float*) * i);
	HostSums = (float**)malloc(sizeof(float*) * i);
	HostSumsSummed = (float**)malloc(sizeof(float*) * i);
	sizes = (int*) malloc(sizeof(int) * i);

	n = num_elements;
	i = 0;
	while ((n = ceil((float)n /  (float)(BLOCK_SIZE*2))) > 1)
	{
		cudaMalloc((void**)&(BlockSums[i]), n * sizeof(float));
		cudaMalloc((void**)&(BlockSumsSummed[i]), n * sizeof(float));
		HostSums[i]= (float*)malloc(n* sizeof(float));
		HostSumsSummed[i]= (float*)malloc(n* sizeof(float));
//		for(int x = 0; x < n; x ++)
//		{
//			HostSums[i][x] = 0.0;
//			HostSumsSummed[i][x] = 0.0;
//		}
//		cudaMemcpy(BlockSums[i], HostSums[i], n * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMemcpy(BlockSumsSummed[i], HostSumsSummed[i], n * sizeof(float), cudaMemcpyHostToDevice);
		sizes[i] = n;
		i++;
	}
	cudaMalloc((void**)&(BlockSums[i]), n * sizeof(float));
	cudaMalloc((void**)&(BlockSumsSummed[i]), n * sizeof(float));
	HostSums[i]= (float*)malloc(n* sizeof(float));
	HostSumsSummed[i]= (float*)malloc(n* sizeof(float));
	sizes[i] = n;
//	for(int x = 0; x < n; x ++)
//	{
//		HostSums[i][x] = 0.0;
//		HostSumsSummed[i][x] = 0.0;
//	}
//	cudaMemcpy(BlockSums[i], HostSums[i], n * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(BlockSumsSummed[i], HostSumsSummed[i], n * sizeof(float), cudaMemcpyHostToDevice);

	
}


void preallocBlockSums_(int num_elements)
{
	

}

// MP4.2 - Device Functions


// MP4.2 - Kernel Functions


__global__ void AdjustIncr (float * arr, float * incr, int n)
{
	if(blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1 < n)
	{
		arr[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1] += incr[blockIdx.x];
		arr[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += incr[blockIdx.x];
	}
	else if (blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 < n)
	{
		arr[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += incr[blockIdx.x];
	}
}


__global__ void prescanArrayKernel(float *outArray, float * inArray, int numElements, float *blockSums)
{

	__shared__ float temp[BLOCK_SIZE * 2 + BLOCK_SIZE/8];
	int tid= threadIdx.x;


	int start = (BLOCK_SIZE * 2) * blockIdx.x;

	int aj, bj;
	aj = tid;
	bj = tid + BLOCK_SIZE;
	int bankOffsetA = CONFLICT_FREE_OFFSET(aj);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bj);

	if(numElements > start + aj)
	{
		temp[aj + bankOffsetA] = inArray[start + aj];
	}
	else 
	{
		temp[aj + bankOffsetA] = 0.0;
	}
	if(numElements > start + bj)
	{
		temp[bj + bankOffsetB] = inArray[start + bj];
	}
	else
	{
		temp[bj + bankOffsetB] = 0.0;
	}

	int offset = 1;
	for (int d = BLOCK_SIZE; d>0; d>>=1)
	{
		__syncthreads();
		if(tid < d)
		{
			int ai = offset * (2 * tid + 1) -1;
			int bi = offset * (2 * tid + 2) -1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);


			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (tid == 0) 
	{
		temp[BLOCK_SIZE * 2 - 1 + CONFLICT_FREE_OFFSET(BLOCK_SIZE * 2 - 1)] = 0;
	}

	for(int d = 1; d < BLOCK_SIZE * 2; d*=2)
	{
		offset >>= 1;
		__syncthreads();
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) -1;
			int bi = offset * (2 * tid + 2) -1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
 		}
	}
	__syncthreads();


	if (numElements > start + aj)
	{
		outArray[start + aj] = temp[aj + bankOffsetA];
	}
	else
	{
		outArray[start + aj] = 0;
	}
	if (numElements > start + bj)
	{
		outArray[start + bj] = temp[bj + bankOffsetB];
	}
	else
	{
		outArray[start + bj] = 0;
	}



	if(tid == 0)
		blockSums[blockIdx.x] = temp[2 * BLOCK_SIZE - 1] + inArray[start + 2 * BLOCK_SIZE - 1];

}

// **===-------- MP4.2 - Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.

void prescanArrayHelper(float *outArray, float *inArray, int numElements, int index)
{
//	printf("starting helper index %d\n", index);
	dim3 dim_block, dim_grid;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = dim_block.z = 1;

	dim_grid.x = ceil((float)(numElements / (float)(dim_block.x * 2)));
	dim_grid.y = dim_grid.z = 1;



	prescanArrayKernel<<<dim_grid,dim_block>>>(outArray, inArray, numElements, BlockSums[index]);

	if (dim_grid.x > 1)
	{
		prescanArrayHelper(BlockSumsSummed[index], BlockSums[index], dim_grid.x, index+1);
		AdjustIncr<<<dim_grid, dim_block>>>(outArray, BlockSumsSummed[index], numElements);

	}
}

void prescanArray(float *outArray, float *inArray, int numElements)
{
	prescanArrayHelper(outArray,inArray, numElements, 0);


}
// **===-----------------------------------------------------------===**


