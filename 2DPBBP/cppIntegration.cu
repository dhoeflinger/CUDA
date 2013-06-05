////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * Host part of the device code.
 * Compiled with Cuda compiler.
 */

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

#define WIN32
// helper functions and utilities to work with CUDA
#include "helper_cuda.h"
#include "helper_functions.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

///////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_odata  memory to process (in and out)
///////////////////////////////////////////////////////////////////////////////
//__global__ void kernel(...)
//{

//}



void gold (float * data, int L , int P, float* img, int N, float Tdet, float cen_det, float dTheta)
{

	for (int px = 0; px < P; px++)
	{
		float theta = dTheta * px;

		for (int lx = 0; lx < L; lx++)
		{
			float rx_comp = -lx * sin(theta);
			for (int ly = 0; ly < L; ly++)
			{
				float ry_comp = ly * cos(theta);

				float tloc = ry_comp + rx_comp;
				img[ly * N + lx] = data[(int)tloc] * ((int)tloc + 1 - tloc)  + data[(int)tloc + 1] * (tloc - (int)tloc);
			
			}

		}
	}


}


////////////////////////////////////////////////////////////////////////////////
//! Entry point for Cuda functionality on host side
////////////////////////////////////////////////////////////////////////////////
extern "C" bool
runTest(float* data, int L, int P, float* img, int N, float Tdet, float cen_det, float dTheta)
{
    findCudaDevice(0, NULL);

	// setup/alloc/copy

    // execute the kernel
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

//    kernel<<< grid, threads >>>(...);
   
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
    printf("Processing time: %.3f msec\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

	// copy results

	// cleanup

    return true;
}
