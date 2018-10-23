/*******************************************************************
 * 
 * itkCuda.h
 * 
 * header file for common includes for both C/C++ and CUDA codes
 *
 * Won-Ki Jeong (wjeong@nvidia.com, wkjeong@cs.utah.edu)
 *
 * June 07, 2007
 *
 *******************************************************************/

#ifndef __ITKCUDACOMMON_H__
#define __ITKCUDACOMMON_H__

//#define DEBUG

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


#ifndef uint
typedef unsigned int uint;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#define MAX_KERNEL_SIZE 200  // max kernel size for convolution filters 
#define MAX_LOCAL_ITER 4     // # of local iteration for finite difference methods

// type definitions

enum FILTERTYPE { FLOAT_TO_FLOAT, 
                  UCHAR_TO_FLOAT, 
                  FLOAT_TO_UCHAR, 
                  UCHAR_TO_UCHAR
                   };

enum FiniteDifferenceFunctionType { GradientAnisotropicDiffusion, 
                                    CurvatureAnisotropicDiffision };

struct AnisotropicDiffusionParameter {
  bool m_GradientMagnitudeIsFixed;
  double m_TimeStep, m_ConductanceParameter, m_FixedAverageGradientMagnitude;
};

// global variables

static FiniteDifferenceFunctionType FDFunctionType;
static AnisotropicDiffusionParameter anisoParam;

inline uint log2(uint i)
{
	assert(i>0);
  uint in  = i;
	uint ret = 0;
	while(i>1)
  {
    ret++;
    //printf("%d, %d\n", i, ret);
    i = i >> 1;
  }
  if(in > (uint)(2 << (ret-1))) ret += 1;
	return ret;
}

#endif
