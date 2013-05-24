/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial compute r software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */




#include "2Dconvolution.h"
#include <cuda.h>

 __constant__ float Md[KERNEL_SIZE * KERNEL_SIZE];
float Mh[KERNEL_SIZE * KERNEL_SIZE];


bool
CopyToConstMem(float * M, size_t msize)
{
	cudaError_t err = cudaMemcpyToSymbol(Md, M, msize);
	return (err == cudaSuccess);

}

// Matrix convolution kernel specification
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{
	const unsigned int dim_size = (BLOCK_SIZE + (KERNEL_SIZE-1)); 
	__shared__ float ds_N[dim_size][dim_size];

	int halosize = (KERNEL_SIZE-1) / 2;

	int ii = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	int I = ii%(dim_size);
	int J = ii/(dim_size);

	int I_o = blockIdx.x * BLOCK_SIZE  + (blockIdx.y * BLOCK_SIZE) * N.width;

	int I_real= I + blockIdx.x * BLOCK_SIZE - halosize;
	int J_real= J + blockIdx.y * BLOCK_SIZE - halosize;

	int I_n = I_o  + (J - halosize)  * (N.width) + I - halosize;


	ds_N[J][I] = (I_real < 0 || I_real >= N.width || J_real < 0 || J_real >= N.height) ? 0 : N.elements[I_n];



	int ii2 = BLOCK_SIZE * BLOCK_SIZE + ii;
	I = ii2%(dim_size);
	J = ii2/(dim_size);

	I_real= I + blockIdx.x * BLOCK_SIZE - halosize;
	J_real= J + blockIdx.y * BLOCK_SIZE - halosize;
	
	I_n = I_o  + (J - halosize)  * (N.width) + I - halosize;

	
	if (ii2 < dim_size*dim_size)
	{
		if (I_real < 0 || I_real >= N.width || J_real < 0 || J_real >= N.height) 
		{
			ds_N[J][I] = 0;
		}
		else
		{
			ds_N[J][I] = N.elements[I_n];
		}
	}

	syncthreads();

	if ((blockIdx.x * BLOCK_SIZE + threadIdx.x < P.width) && (blockIdx.y * BLOCK_SIZE + threadIdx.y < P.height))
	{
		for (unsigned int j = 0; j < KERNEL_SIZE; j++)
		{
			for (unsigned int i = 0; i < KERNEL_SIZE; i++)
			{
				
				P.elements[blockIdx.x * BLOCK_SIZE + threadIdx.x + (blockIdx.y * BLOCK_SIZE + threadIdx.y) * P.width] += Md[i + j * KERNEL_SIZE] * ds_N[j + threadIdx.y][i + threadIdx.x];
			}
		}
	}

}