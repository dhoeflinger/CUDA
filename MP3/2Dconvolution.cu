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
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Matrix convolution.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <fstream>
#include <sstream>
#include <iostream>

// includes, kernels
#include "2Dconvolution.h"
#include <cuda_runtime.h>


////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
int ReadFile(float* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);


void DiffMatrix (float * M, float * N, float * res, size_t size_of_array)
{
	for (unsigned int x = 0; x < size_of_array; x ++)
	{
		float diff= fabs(M[x] - N[x]);
		res[x] = diff < 0.001f ? 0 : diff; 
	}


}

bool CompareMatrix(float * M, float * N,  size_t size_of_array, float threshold)
{	
	bool passed = true;
	for (unsigned int x = 0; x < size_of_array; x ++)
	{
		passed &= (fabs(M[x] - N[x]) < threshold);
	}
	return passed;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	
	srand(2012);
	
	if(argc != 5 && argc != 4) 
	{
		// Allocate and initialize the matrices
		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 1); 
		N  = AllocateMatrix((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
		P  = AllocateMatrix(N.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = (int*)malloc(3 * sizeof(int));
		
		unsigned int data_read = 0;
		std::ifstream source(argv[1], std::ios_base::in);
		source >> params[0];
		source >> params[1];
		source >> params[2];

		//		cutReadFilei(argv[1], &params, &data_read, true);
		if(data_read != 2){
			printf("Error reading parameter file\n");
			free(params);
			return 1;
		}
		M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE, 0); 
		N  = AllocateMatrix(params[0], params[1], 0);		
		P  = AllocateMatrix(params[0], params[1], 0);
		free(params);

		(void)ReadFile(&M, argv[2]);
		(void)ReadFile(&N, argv[3]);
	}

	// M * N on the device
    ConvolutionOnDevice(M, N, P);
    
    // compute the matrix convolution on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    computeGold(reference.elements, M.elements, N.elements, N.height, N.width);
        
    // in this case check if the result is equivalent to the expected soluion
    bool res = CompareMatrix(reference.elements, P.elements, P.width * P.height, 0.001f);

	Matrix diff = AllocateMatrix(P.height, P.width, 0);
	DiffMatrix (reference.elements, P.elements, diff.elements, P.width * P.height);

	WriteFile(diff, "diff.out");

	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	WriteFile(P, "gpu.out");

WriteFile(N, "n.out");

	WriteFile(reference, "gold.out");

    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}

////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////

	CopyToConstMem(M.elements, 	KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Setup the execution configuration
	dim3 dim_block, dim_grid;

	dim_block.x = dim_block.y = BLOCK_SIZE;
	dim_block.z = 1;

	dim_grid.x = ceil((float)(P.width / (float)dim_block.x));
	dim_grid.y = ceil((float)(P.height / (float)dim_block.y));
	dim_grid.z = 1;


    // Launch the device computation threads!
	ConvolutionKernel<<<dim_grid,dim_block>>>(Md, Nd, Pd);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
//    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
		if(rand() % 2)
			M.elements[i] = - M.elements[i];
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height * M->width;
	std::ifstream source(file_name, std::ios_base::in);

	for (unsigned int x = 0; x < data_read; x++)
	{
		source >> M->elements[x];
	}


	return data_read;
}

int ReadFile(float* M, char* file_name)
{
	unsigned int data_read = KERNEL_SIZE * KERNEL_SIZE;
	std::ifstream source(file_name, std::ios_base::in);

	for (unsigned int x = 0; x < data_read; x++)
	{
		source >> M[x];
	}


	return data_read;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
	unsigned int data_write = M.width * M.height;

	std::ofstream dest(file_name, std::ios_base::out);

	for (unsigned int x = 0; x < data_write; x++)
	{
		dest << M.elements[x]<<" ";
	}
}
