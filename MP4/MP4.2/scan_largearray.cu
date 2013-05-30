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

#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include <fstream>
#include <sstream>
#include <iostream>

#define WIN32
#include "helper_timer.h"

#include "scan_largearray_kernel.h"

// includes, kernels


#define DEFAULT_NUM_ELEMENTS 16500000
#define MAX_RAND 3




////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(float*, char* file_name, int size);
void WriteFile(float*, char* file_name, int size);

extern "C" 
unsigned int compare( const float* reference, const float* data, 
                     const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);


bool Compare(float * M, float * N,  size_t size_of_array, float threshold)
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
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int errorM = 0;
    float device_time;
    float host_time;
    int* size = NULL; //(int*)malloc(1 * sizeof(int));
    unsigned int data2read = 1;
    int num_elements = 0; // Must support large, non-power-of-2 arrays

    // allocate host memory to store the input data
    unsigned int mem_size = sizeof( float) * num_elements;
    float* h_data = (float*) malloc( mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Randomly generate input data and write the result to
    //   file name specified by first argument
    // * Two arguments: Read the first argument which indicate the size of the array,
    //   randomly generate input data and write the input data
    //   to the second argument. (for generating random input data)
    // * Three arguments: Read the first file which indicate the size of the array,
    //   then input data from the file name specified by 2nd argument and write the
    //   SCAN output to file name specified by the 3rd argument.
    switch(argc-1)
    {      
        case 2: 
            // Determine size of array		
			{
				std::ifstream source(argv[1], std::ios_base::in);
				source >> data2read;
				
				if(data2read != 1){
					printf("Error reading parameter file\n");
					exit(1);
				}
				
				num_elements = size[0];
				
				// allocate host memory to store the input data
				mem_size = sizeof( float) * num_elements;
				h_data = (float*) malloc( mem_size);
				
				for( unsigned int i = 0; i < num_elements; ++i)
				{
                h_data[i] = (int)(rand() % MAX_RAND);
				}
				WriteFile(h_data, argv[2], num_elements);
			}
        break;
    
        case 3:			// Three Arguments
			{
				std::ifstream source(argv[1], std::ios_base::in);
				source >> data2read;
				
				if(data2read != 1){
					printf("Error reading parameter file\n");
					exit(1);
				}
				
				num_elements = size[0];
				
				// allocate host memory to store the input data
				mem_size = sizeof( float) * num_elements;
				h_data = (float*) malloc( mem_size);
				
				errorM = ReadFile(h_data, argv[2], size[0]);
				if(errorM != 1)
				{
					printf("Error reading input file!\n");
					exit(1);
				}
			}
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            // Use DEFAULT_NUM_ELEMENTS num_elements
            num_elements = DEFAULT_NUM_ELEMENTS;
            
            // allocate host memory to store the input data
            mem_size = sizeof( float) * num_elements;
            h_data = (float*) malloc( mem_size);

            // initialize the input data on the host
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
//                h_data[i] = 1.0f;
                h_data[i] = (int)(rand() % MAX_RAND);
            }
        break;  
    }    

    

	StopWatchWin timer;


      
    // compute reference solution
    float* reference = (float*) malloc( mem_size);  
	timer.start();
    computeGold( reference, h_data, num_elements);
	timer.stop();

    printf("\n\n**===-------------------------------------------------===**\n");
    printf("Processing %d elements...\n", num_elements);
	printf("Host CPU Processing time: %f (ms)\n", timer.getTime());

	host_time = timer.getTime();



    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;

	cudaMalloc( (void**) &d_idata, mem_size);
	cudaMalloc( (void**) &d_odata, mem_size);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice);
    // initialize all the other device arrays to be safe
    cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice);

    // **===-------- MP4.2 - Allocate data structure here -----------===**
	preallocBlockSums(num_elements);
    // **===-----------------------------------------------------------===**

    // Run just once to remove startup overhead for more accurate performance 
    // measurement
    prescanArray(d_odata, d_idata, 16);

    // Run the prescan
	timer.reset();
	timer.start();
    
    // **===-------- MP4.2 - Modify the body of this function -----------===**
    prescanArray(d_odata, d_idata, num_elements);
    // **===-----------------------------------------------------------===**
    cudaThreadSynchronize();

	timer.stop();

	printf("G80 CUDA Processing time: %f (ms)\n", timer.getTime());
	device_time = timer.getTime();
    printf("Speedup: %fX\n", host_time/device_time);

    // **===-------- MP4.2 - Deallocate data structure here -----------===**
    // deallocBlockSums();
    // **===-----------------------------------------------------------===**


    // copy result from device to host
    cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, 
                               cudaMemcpyDeviceToHost);

    if ((argc - 1) == 3)  // Three Arguments, write result to file
    {
        WriteFile(h_data, argv[3], num_elements);
    }
    else if ((argc - 1) == 1)  // One Argument, write result to file
    {
        WriteFile(h_data, argv[1], num_elements);
    }


    // Check if the result is equivalent to the expected soluion
    unsigned int result_regtest = Compare( reference, h_data, num_elements, 0.0001f);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");

    // cleanup memory
    free( h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
}


//int ReadFile(float* M, char* file_name, int size)
//{
//	unsigned int elements_read = size;
//	if (cutReadFilef(file_name, &M, &elements_read, true))
//        return 1;
//    else
//        return 0;
//}

//void WriteFile(float* M, char* file_name, int size)
//{
//    cutWriteFilef(file_name, M, size, 0.0001f);
//}
//



// Read a 16x16 floating point matrix in from file
int ReadFile(float* M, char* file_name, int size)
{
	unsigned int data_read = size;
	std::ifstream source(file_name, std::ios_base::in);

	for (unsigned int x = 0; x < data_read; x++)
	{
		source >> M[x];
	}

		return 1;
}



// Write a 16x16 floating point matrix to file
void WriteFile(float * M, char* file_name, int size)
{
	unsigned int data_write = size;

	std::ofstream dest(file_name, std::ios_base::out);

	for (unsigned int y = 0; y < data_write; y++)
	{
			dest << M[y]<<" ";
	}
}
