ECE408 Machine Problem 3
Blocked 2D Convolution

This MP is a blocked implementation of a matrix convolution. This assignment
will have a constant 5x5 convolution kernel, but will have arbitrarily sized
"images".  

Matrix convolution is primarily used in image processing for tasks such as
image enhancing, blurring, etc. A standard image convolution formula for a 5x5
convolution filter M with an Image N is:

P(i,j) = sum (a = 0 to 4) { sum(b = 0 to 4) { M[a][b] * N[i+a-2][j+b-2] } }
where 0 <= i < N.height and 0 <= j < N.width

Elements that are "outside" Matrix N, for this MP, are treated as if they
had the value zero. 

1) Unpack the tarball into your SDK "C/src" directory.

2) Edit the source files "2Dconvolution.cu" and "2Dconvolution_kernel.cu" to
   complete the functionality of matrix convolution on the device. Compiling
   your code will produce the executable mp3-convolution in your SDK
   "C/bin/linux/release" directory.

3) There are several modes of operation for the application:  

   No arguments: the application will create a randomized Filter M and Image N.
   A CPU implementation of the convolution algorithm will be used to generate 
   a correct solution which will be compared with your program's output. If it 
   matches (within a certain tolerance), it will print out "Test PASSED" to the
   screen before exiting.

   One argument: the application will create a randomized Filter M and Image N,
   and write the device-computed output to the file specified by the argument.  

   Three arguments: the application will read the filter and image from 
   provided files. The first argument should be a file containing two integers
   representing the image height and width respectively. The second and third
   function arguments should be files which have exactly enough entries to fill
   the Filter M and Image N respectively. No output is written to file.  
   
   Four arguments: the application will read its inputs using the files
   provided by the first three arguments, and write its output to the file
   provided in the fourth.  

   Note that if you wish to use the output of one run of the application as an
   input, you must delete the first line in the output file which displays the
   accuracy of the values within the file. The value is not relevant for this
   application.

4) Report:

   It's time to do some performance testing and analysis. Included in the 
   MP3-convolution folder is a folder called "test", which contains two test
   case input sets. Using these test cases, and any others that you wish to
   create to support your findings, answer the following questions, along with
   a short description of how you arrived at those answers.

   1. What is the measured floating-point computation rate for the CPU and GPU
      kernels in this application? How do they each scale with the size of the
      input?

   2. How much time is spent as an overhead cost for using the GPU for 
      computation? Consider all code executed within your host function
      ConvolutionOnDevice, with the exception of the kernel itself, as overhead.
      How does the overhead scale with the size of the input?

   You are free to use any timing library you like, as long as it has
   a reasonable accuracy. Note that the CUDA utility library "cutil" provides
   some timing functions if you care to use those. Its header file is in your
   SDK "C/common/inc" directory.

   Remember that kernel invocations are normally asynchronous, so if you want
   accurate timing of the kernel's running time, you need to insert a call to
   "cudaDeviceSynchronize()" after the kernel invocation.  

5) Submit your solution via Compass as a tarball. Your submission should
   contain the "MP3-convolution" folder provided, with all the changes and
   additions you made to the source code. In addition to the folder, add a text 
   file, Word Document, or PDF file with your answers to the questions.


Grading:  

Your submission will be graded on the following parameters:

Functionality/knowledge: 65%
    - Produces correct output results.
    - Handles boundary conditions correctly.
    - Uses shared, constant or texture memory appropriately to cover global
      memory access latency.

Report: 35%

