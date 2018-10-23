================
CUDA ITK add-ons
================

This is CUDA implementation of well-known ITK image filters.

0. System requirements
- MS Windows XP or linux
- NVIDIA GeForce 8 series or Quadro 5600/4600 graphcis card
- NVIDIA CUDA 2.2 
- ITK 3.14.0
- CMake 2.6 or later

1. Installation

Copy all files (including subdirectories) into the root of ITK source tree (can be downloaded from www.itk.org). By doing so, some ITK source files will be replaced with modified ones using CUDA.

2. Compilation

Run CMake and create VC project file. The process is same as regular ITK compilation.

3. How to use

Link itkCuda.lib where you need. For example, add the following in CMakeLists : TARGET_LINK_LIBRARIES(your-program itkCuda). When your code is executed, set the global variable as follows to choose CPU/GPU itk code at runtime:

set ITK_CUDA = 0 (original itk)
set ITK_CUDA = 1 (CUDA itk)

4. Timing

Set the global variable CUDA_LAUNCH_BLOCKING=1 for accurate timing for GPU code (for CUDA 1.0 or later)


======
NOTE!!
======

This code is evaluation purpose only and there is no guarantee at all. This code is to demonstrate how NVIDIA CUDA can be integrated into existing ITK library for GPU processing. This code should not be used for other than testing / educational purpose.


==========
Known bugs
==========

1. Error near boundary of the image for anisotropic filter (boundary condition problem.. will be fixed later but no time right now)
2. Results of anisotropic diffusion filter are different from original ITK and CUDA version (due to different implementation of discretization)
3. Large kernel (e.g., 11^3) would not work for mean filter due to the shared memory limitation


================
Acknowledgements
================

I would like to thank to Thomas Wieland from UMIT in tyrol, Austria for helping me to test code on Linux and modify CMake script. CudaAtlasImageCreator is also written by him.



Contact me if you have any questions.

Won-Ki Jeong (wkjeong@seas.harvard.edu)

2007. 8. 17
Modified on 2007. 9. 12
Modified on 2009. 4. 8
Modified on 2009. 4. 14
Modified on 2009  4. 16
Modified on 2009. 6. 27

Copyright (c) 2007, 2008, 2009 Won-Ki Jeong