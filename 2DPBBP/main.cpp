/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 * Compiled with default CPP compiler.
 */

// includes, system
#include <iostream>
#include <stdlib.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>

#include "IRUtility/Data/Array.hpp"
#include "IRUtility/Oper/Timer.hpp"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" bool runTest(float* data, int L, int P, float* img, int N, float Tdet, float cen_det, float dTheta);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
	Array2D<float> a;
	a.Deserialize("thorax_par_projections.dat");

	Array2D<float> b;
	b.Allocate(512, 512);

	float Tdet = 1;
	float cen_det = 367;
	float dTheta = 0.0040906154;

	Timer full_timer;
	full_timer.Start();
    // run the device part of the program
    bool bTestResult = runTest(a[0], a.GetRowLength(), a.GetRowCount(), b[0], 512, Tdet, cen_det, dTheta);
	full_timer.Stop();
	printf("Run took %.2f sec\n", full_timer.TimeElapsed());

    cudaDeviceReset();

	b.Serialize("recon.img");

    return bTestResult ? EXIT_SUCCESS : EXIT_FAILURE;
}
