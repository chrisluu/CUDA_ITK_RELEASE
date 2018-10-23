/*******************************************************************
 * itkCuda.h
 * 
 * Header file for itkCuda library. Include this to use itkCuda library
 *
 * Environment variables :
 * If you want to run CUDA filter, set ITK_CUDA = 1
 * If you want to check running time, set ITK_CUDA_TIME = 1
 *
 * Won-Ki Jeong (wjeong@nvidia.com, wkjeong@cs.utah.edu)
 *
 * June 07, 2007
 *
 *******************************************************************/

#ifndef __ITKCUDA_H__
#define __ITKCUDA_H__

//#define DEBUG

// ITK CUDA COMMON
#include "itkCudaCommon.h"

// ITKCommon
#include "itkSize.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkNthElementImageAdaptor.h"
#include "itkDerivativeOperator.h"
#include "itkGaussianOperator.h"
#include "itkImageRegionIteratorWithIndex.h"


extern "C"
void cudaMeanImageFilterKernelWrapper(void *_in, void *_out,
                                      FILTERTYPE type, int dimension,
                                      uint image_width, uint image_height, uint image_depth,
                                      uint kernel_width, uint kernel_height, uint kernel_depth);
                                  
//
// Out-Of-Core implementation
//
extern "C"
void cudaMeanImageFilterKernelWrapperOFC(void *_in, void *_out,
                                         FILTERTYPE type, int dimension,
                                         uint image_width, uint image_height, uint image_depth,
                                         uint kernel_width, uint kernel_height, uint kernel_depth,
                                         uint offset_x, uint offset_y, uint offset_z,
                                         uint logical_width, uint logical_height, uint logical_depth);

extern "C"
void cudaMedianImageFilterKernelWrapper(void *_in, void *_out,
                                        FILTERTYPE type, int dimension,
                                        uint image_width, uint image_height, uint image_depth,
                                        uint kernel_width, uint kernel_height, uint kernel_depth);

extern "C"
void cudaConvolutionFilter1DKernelWrapper(void *_in, void *_out, float *kernel,
                                          FILTERTYPE type, int indir, int outdir, int kernelsize, 
                                          uint image_width, uint image_height, uint image_depth);
/*
// previous version without using reordering in/out global memory - slow for z axis
extern "C"
void cudaConvolutionFilter1DKernelWrapper(void *_in, void *_out, float *kernel,
                                          FILTERTYPE type, int direction, int kernelsize, 
                                          uint image_width, uint image_height, uint image_depth);
*/

extern "C"
void cudaGradientAnisotropicDiffusionFilterKernelWrapper(void *_in, void *_out,
                                                         FILTERTYPE type, int dimension, int numiter,
                                                         AnisotropicDiffusionParameter param,
                                                         uint image_width, uint image_height, uint image_depth);


namespace itk
{

// Global functions
void initCUDA();

inline bool checkCUDA()
{
  bool _cuda = false;
  char *cudavar = getenv("ITK_CUDA");
  if(cudavar != NULL)
	{
		if(cudavar[0] == '1') 
		{
      _cuda = true;
			//std::cout << "ITK CUDA ENABLED" << std::endl;
		}
	}
  return _cuda;
}

inline bool checkTime()
{
  bool _cuda = false;
  char *cudavar = getenv("ITK_CUDA_TIME");
  if(cudavar != NULL)
	{
		if(cudavar[0] == '1') 
		{
      _cuda = true;
			//std::cout << "ITK CUDA ENABLED" << std::endl;
		}
	}
  return _cuda;
}


//
// CUDA Mean Filter for itk
//
template <class InputImageType, class OutputImageType, class InputSizeType>
void cudaMeanImageFilter(const InputImageType *in, OutputImageType *out, 
                            InputSizeType m_radius)
{
  typedef typename InputImageType::SizeType sizetype;

  // out-of-core version
  int w, h, d, dim;
  dim = in->GetImageDimension();

  // ldim : logical input dimension
  // pdim : physical input dimension
  sizetype imageSize = in->GetLargestPossibleRegion().GetSize();
  dim = in->GetImageDimension();
  w = imageSize[0];
  h = imageSize[1];
  d = imageSize[2];
  
  sizetype min_off, max_off; // min/max offsets along x/y/z axis
  min_off[0] = min_off[1] = max_off[0] = max_off[1] = 0;

  typename InputImageType::PixelType *d_mem[2];
 
  // get total GPU memory
  uint gpumemsize;
  cuDeviceTotalMem(&gpumemsize, 0);
  int nPiece = ceil((float)(imageSize[0]*imageSize[1]*imageSize[2]*(sizeof(typename InputImageType::PixelType)+sizeof(typename OutputImageType::PixelType)))/(float)gpumemsize);

  std::cout << "# of pieces : " << nPiece << std::endl;

  unsigned int strMemLoc, inputsize, outputsize;
  for(int i=0; i<nPiece; i++)
  {
    if(nPiece == 1)
    {
      d = imageSize[2];
      min_off[2] = max_off[2] = 0;
      strMemLoc = 0;     
    }
    else
    {
      if(i==0)
      {        
        d = imageSize[2]/nPiece + m_radius[2];
        min_off[2] = 0;
        max_off[2] = m_radius[2];    
        strMemLoc = 0;
      }
      else if(i < nPiece-1)
      {
        d = imageSize[2]/nPiece + 2*m_radius[2]; 
        min_off[2] = max_off[2] = m_radius[2]; 
        strMemLoc = (imageSize[2]/nPiece)*i - m_radius[2];
      }
      else 
      {
        d = imageSize[2] - i*(imageSize[2]/nPiece) + m_radius[2];
        min_off[2] = m_radius[2];
        max_off[2] = 0;
        strMemLoc = (imageSize[2]/nPiece)*i - m_radius[2];
      }
    }
    inputsize  = sizeof(typename InputImageType::PixelType)*w*h*d;
    outputsize = sizeof(typename OutputImageType::PixelType)*w*h*(d - min_off[2] - max_off[2]);
   
    cudaMalloc((void**) &d_mem[0], inputsize);
    cudaMalloc((void**) &d_mem[1], inputsize);
    CUT_CHECK_ERROR("Memory creation failed");

    cudaMemcpy(d_mem[0], in->GetBufferPointer() + w*h*strMemLoc , inputsize, cudaMemcpyHostToDevice);
    CUT_CHECK_ERROR("Memory copy failed");

    // call filter function
    int kw, kh, kd; // Mean filter kernel width/height/depth
    kw = kh = kd = 0;
    if(dim > 0) kw = m_radius[0];
    if(dim > 1) kh = m_radius[1];
    if(dim > 2) kd = m_radius[2];  
    
    if(sizeof(typename InputImageType::PixelType)==1)
    {
      cudaMeanImageFilterKernelWrapperOFC((typename InputImageType::PixelType*)d_mem[0], 
                                          (typename OutputImageType::PixelType*)d_mem[1], 
                                          UCHAR_TO_UCHAR, dim, w, h, d, kw, kh, kd,
                                          min_off[0], min_off[1], min_off[2], 
                                          max_off[0], max_off[1], max_off[2]);
    }
    else
    {
      cudaMeanImageFilterKernelWrapperOFC((typename InputImageType::PixelType*)d_mem[0], 
                                          (typename OutputImageType::PixelType*)d_mem[1], 
                                          FLOAT_TO_FLOAT, dim, w, h, d, kw, kh, kd,
                                          min_off[0], min_off[1], min_off[2], 
                                          max_off[0], max_off[1], max_off[2]);
    }
    
    cudaMemcpy(out->GetBufferPointer() + w*h*(strMemLoc + min_off[2]), d_mem[1] + w*h*min_off[2], outputsize, cudaMemcpyDeviceToHost);
    CUT_CHECK_ERROR("Memory copy failed");

    // Free device memory
    cudaFree(d_mem[0]);
    cudaFree(d_mem[1]);
  }
}



/*
//
// CUDA Mean Filter for itk
//
template <class InputImageType, class OutputImageType, class InputSizeType>
void cudaMeanImageFilter(const InputImageType *in, OutputImageType *out, InputSizeType m_radius)
{

  typedef typename InputImageType::SizeType sizetype;

#ifdef DEBUG
  unsigned int timer[3];
  timer[0] = timer[1] = timer[2] = 0;
  CUT_SAFE_CALL(cutCreateTimer(&(timer[0])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[1])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[2])));
#endif

  int w, h, d, dim;

  sizetype imageSize = in->GetLargestPossibleRegion().GetSize();
  dim = in->GetImageDimension();
  w = imageSize[0];
  h = imageSize[1];
  d = imageSize[2];

  // Dimension of the input image should be either 2 or 3
  if(dim < 2 || dim > 3)
  {
    std::cerr << "cudaMeanImageFilter only accepts 2/3D images" << std::endl;
    assert(false);
  } 

  // Setting block size
  if(dim == 2) d = 1;

  std::cout << "Image Dimension : " << dim << std::endl;
  std::cout << "Image Size : " << w << "x" << h << "x" << d << std::endl;
  std::cout << "Size of pixel type : " << sizeof(InputImageType::PixelType) << std::endl;
  
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif
  // Create host/device memory
  InputImageType::PixelType *d_mem[2];  
  cudaMalloc((void**) &d_mem[0], sizeof(InputImageType::PixelType)*w*h*d);
  cudaMalloc((void**) &d_mem[1], sizeof(InputImageType::PixelType)*w*h*d);
  CUT_CHECK_ERROR("Memory creation failed");
 
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
#endif

  // copy host to device memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif
  cudaMemcpy(d_mem[0], in->GetBufferPointer(), sizeof(InputImageType::PixelType)*w*h*d, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy failed");
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif

  // call filter function
  int kw, kh, kd; // Mean filter kernel width/height/depth
  kw = kh = kd = 0;
  if(dim > 0) kw = m_radius[0];
  if(dim > 1) kh = m_radius[1];
  if(dim > 2) kd = m_radius[2];  

#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[0]));
#endif

  // type check -- should be better than this but will be fixed later
  if(sizeof(InputImageType::PixelType)==1)
  {
    cudaMeanImageFilterKernelWrapper((InputImageType::PixelType*)d_mem[0], 
                                     (OutputImageType::PixelType*)d_mem[1], 
                                     UCHAR_TO_UCHAR, dim, w, h, d, kw, kh, kd);
  }
  else
  {
    cudaMeanImageFilterKernelWrapper((InputImageType::PixelType*)d_mem[0], 
                                     (OutputImageType::PixelType*)d_mem[1], 
                                     FLOAT_TO_FLOAT, dim, w, h, d, kw, kh, kd);
  }

 
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[0]));
#endif
  
  // copy device to host memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif

  cudaMemcpy(out->GetBufferPointer(), d_mem[1], sizeof(OutputImageType::PixelType)*w*h*d, cudaMemcpyDeviceToHost);
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif

  // copy host to itkImage
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif
 
  // Free device memory
  cudaFree(d_mem[0]);
  cudaFree(d_mem[1]);

#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
 
  printf("Kernel Time: %f (ms)\n", cutGetTimerValue(timer[0]));
  printf("CPU <-> CPU Time: %f (ms)\n", cutGetTimerValue(timer[1]));
  printf("CPU <-> GPU Time: %f (ms)\n", cutGetTimerValue(timer[2]));
  
  CUT_SAFE_CALL(cutDeleteTimer(timer[0]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[1]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[2]));
#endif
}
*/


//
// CUDA Discrete Gaussian Filter for itk
//
// GPU memory usage :
// Minimum 2*(w*h*d*sizeof(float)) + kernel_size*sizeof(float) is required 
//
template <class InputImageType, class OutputImageType, class OperatorType>
void cudaDiscreteGaussianImageFilter(const InputImageType *in, OutputImageType *out, 
                                     std::vector<OperatorType> oper)
{
  typedef typename InputImageType::PixelType InPixType;
  typedef typename OutputImageType::PixelType OutPixType;
  
#ifdef DEBUG
  unsigned int timer[3];
  timer[0] = timer[1] = timer[2] = 0;
  CUT_SAFE_CALL(cutCreateTimer(&(timer[0])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[1])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[2])));
#endif

  int w, h, d, dim;
 
  typename InputImageType::SizeType imageSize = in->GetLargestPossibleRegion().GetSize();
  dim = in->GetImageDimension();

  // Dimension of the input image should be either 2 or 3
  if(dim < 1 || dim > 3)
  {
    std::cerr << "cudaMeanImageFilter only accepts 1/2/3D images" << std::endl;
    assert(false);
  } 

  w = imageSize[0];
  if(dim == 1) 
  {
    h = d = 1;
  }
  else if(dim == 2)
  { 
    h = imageSize[1];
    d = 1;
  }
  else
  {
    h = imageSize[1];
    d = imageSize[2];
  }

#ifdef DEBUG
  // debug : print out Gaussian kernel
  for(int i=0; i<dim; i++)
  {
    printf("Kernel for dim %d\n", i+1);
    double sum = 0;
    for(int j=0; j<oper[i].GetSize(i); j++)
    {
      double val = ((oper[i].GetBufferReference()))[j];
      sum += val;
      printf("%f ", val);
    }
    printf("\nSum of kernel : %f\n\n", sum);
  }

  std::cout << "Image Dimension : " << dim << std::endl;
  std::cout << "Image Size : " << w << "x" << h << "x" << d << std::endl;
  std::cout << "Size of pixel type : " << sizeof(InPixType) << std::endl;

  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif
  // Create host/device memory
  int kernel_size[3];
  float* h_kernel[3];
  float* d_temp[2];  
  InPixType *d_in;
  OutPixType *d_out;
 
  // create Gaussian kernels
  for(int i=0; i<dim; i++)
  {
    kernel_size[i] = oper[i].GetSize(i);

    assert(kernel_size[i] <= MAX_KERNEL_SIZE);

    h_kernel[i] = (float*)malloc(sizeof(float)*(kernel_size[i]));
  
    // copy Gaussian kernel weight
    for(int j=0; j<kernel_size[i]; j++)
    {
      h_kernel[i][j] = oper[i][j];   
    }  
  }

  CUT_CHECK_ERROR("Memory creation failed");
 
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
#endif

  // copy host to device memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif
 
  cudaMalloc((void**) &d_in,  sizeof(InPixType)*w*h*d);  CUT_CHECK_ERROR("Memory creation failed");
  cudaMemcpy(d_in, in->GetBufferPointer(), sizeof(InPixType)*w*h*d, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy failed");

#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif


#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[0]));
#endif

  FILTERTYPE filterType;

  if(dim == 1)
  {  
    // filtering along x
    if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 1) filterType = UCHAR_TO_UCHAR;
    else if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 4) filterType = UCHAR_TO_FLOAT;
    else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 1) filterType = FLOAT_TO_UCHAR;
    else
    {
      printf("No such filter type\n");
      assert(false); exit(0);
    }

    cudaMalloc((void**) &d_out, sizeof(OutPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
 
    cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                         (OutPixType*)d_out,
                                          h_kernel[0],
                                          filterType, 0, 0,
                                          oper[0].GetSize(0),
                                          w, h, d);

    cudaFree((void*)d_in);  CUT_CHECK_ERROR("Memory free failed");

  }
  else if(dim == 2)
  { 
    // filtering along x
    if(sizeof(InPixType) == 1) filterType = UCHAR_TO_FLOAT;
    else if(sizeof(InPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type\n");
      assert(false); exit(0);
    }   

    cudaMalloc((void**) &d_temp[0], sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");

    cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                         (float*)d_temp[0],
                                          h_kernel[0],
                                          filterType, 0, 1,
                                          oper[0].GetSize(0), w, h, d);

    cudaFree((void*)d_in); CUT_CHECK_ERROR("Memory free failed");

    // filtering along y
    if(sizeof(OutPixType) == 1) filterType = FLOAT_TO_UCHAR;
    else if(sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type!\n");
      assert(false); exit(0);
    }

    cudaMalloc((void**) &d_out, sizeof(OutPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
 
    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[0],
                                         (OutPixType*)d_out,
                                          h_kernel[1],
                                          filterType, 1, 0,
                                          oper[1].GetSize(1), w, h, d);  

    cudaFree((void*)d_temp[0]); CUT_CHECK_ERROR("Memory free failed");
  }
  else if(dim == 3)
  { 
    /*
    // filtering along x
    if(sizeof(InPixType) == 1) filterType = UCHAR_TO_FLOAT;
    else if(sizeof(InPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type\n");
      assert(false); exit(0);
    }   
    cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                                 (float*)d_temp[0],
                                                  h_kernel[0],
                                                  filterType, 0, 
                                                  oper[0].GetSize(0), w, h, d);
    // filtering along y
    filterType = FLOAT_TO_FLOAT;    
    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[0],
                                                 (float*)d_temp[1],
                                                  h_kernel[1],
                                                  filterType, 1, 
                                                  oper[1].GetSize(1), w, h, d);
    // filtering along z
    if(sizeof(OutPixType) == 1) filterType = FLOAT_TO_UCHAR;
    else if(sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type!\n");
      assert(false); exit(0);
    }
    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[1],
                                                 (OutPixType*)d_out,
                                                  h_kernel[2],
                                                  filterType, 2, 
                                                  oper[2].GetSize(2), w, h, d);
    */

    // filtering along x
    if(sizeof(InPixType) == 1) filterType = UCHAR_TO_FLOAT;
    else if(sizeof(InPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type\n");
      assert(false); exit(0);
    }   

    cudaMalloc((void**) &d_temp[0], sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");

    cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                         (float*)d_temp[0],
                                          h_kernel[0],
                                          filterType, 0, 2,
                                          oper[0].GetSize(0), w, h, d);

    cudaFree((void*)d_in); CUT_CHECK_ERROR("Memory free failed");
  
    // filtering along z
    filterType = FLOAT_TO_FLOAT;  

    cudaMalloc((void**) &d_temp[1], sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");

    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[0],
                                         (float*)d_temp[1],
                                          h_kernel[2],
                                          filterType, 2, 1,
                                          oper[2].GetSize(2), w, h, d);

    cudaFree((void*)d_temp[0]); CUT_CHECK_ERROR("Memory free failed");
   
    // filtering along y
    if(sizeof(OutPixType) == 1) filterType = FLOAT_TO_UCHAR;
    else if(sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type!\n");
      assert(false); exit(0);
    }

    cudaMalloc((void**) &d_out, sizeof(OutPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
 
    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[1],
                                         (OutPixType*)d_out,
                                          h_kernel[1],
                                          filterType, 1, 0,
                                          oper[1].GetSize(1), w, h, d);

    cudaFree((void*)d_temp[1]); CUT_CHECK_ERROR("Memory free failed");
  }
  else
  {
     printf("Only 1/2/3D input image is supported!\n");
     assert(false); exit(0);
  }


#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[0]));
#endif
  
  // copy device to host memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif

  cudaMemcpy(out->GetBufferPointer(), d_out, sizeof(OutPixType)*w*h*d, cudaMemcpyDeviceToHost);
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif

  // copy host to itkImage
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif

  // free dynamically allocated memory
  cudaFree((void*)d_out); CUT_CHECK_ERROR("Memory free failed");
  for(int i=0; i<dim; i++) free(h_kernel[i]);    
 
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
 
  printf("Kernel Time: %f (ms)\n", cutGetTimerValue(timer[0]));
  printf("CPU <-> CPU Time: %f (ms)\n", cutGetTimerValue(timer[1]));
  printf("CPU <-> GPU Time: %f (ms)\n", cutGetTimerValue(timer[2]));
  
  CUT_SAFE_CALL(cutDeleteTimer(timer[0]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[1]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[2]));
#endif
}


//
// CUDA 1D Derivative Image Filter for ITK
//
template <class InputImageType, class OutputImageType, class OperatorType>
void cudaDerivativeImageFilter(const InputImageType *in, OutputImageType *out, OperatorType& oper)
{
  typedef typename InputImageType::PixelType InPixType;
  typedef typename OutputImageType::PixelType OutPixType;
  
#ifdef DEBUG
  unsigned int timer[3];
  timer[0] = timer[1] = timer[2] = 0;
  CUT_SAFE_CALL(cutCreateTimer(&(timer[0])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[1])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[2])));
#endif

  int w, h, d, dim;
 
  typename InputImageType::SizeType imageSize = in->GetLargestPossibleRegion().GetSize();
  dim = in->GetImageDimension();

  // Dimension of the input image should be either 2 or 3
  if(dim < 1 || dim > 3)
  {
    std::cerr << "cudaDerivativeImageFilter only accepts 1/2/3D images" << std::endl;
    assert(false);
  } 

  w = imageSize[0];
  if(dim == 1) 
  {
    h = d = 1;
  }
  else if(dim == 2)
  { 
    h = imageSize[1];
    d = 1;
  }
  else
  {
    h = imageSize[1];
    d = imageSize[2];
  }

#ifdef DEBUG
  // debug : print out Gaussian kernel
  for(int i=0; i<dim; i++)
  {
    printf("Kernel for dim %d\n", i+1);
    double sum = 0;
    for(int j=0; j<oper[i].GetSize(i); j++)
    {
      double val = ((oper[i].GetBufferReference()))[j];
      sum += val;
      printf("%f ", val);
    }
    printf("\nSum of kernel : %f\n\n", sum);
  }

  std::cout << "Image Dimension : " << dim << std::endl;
  std::cout << "Image Size : " << w << "x" << h << "x" << d << std::endl;
  std::cout << "Size of pixel type : " << sizeof(InPixType) << std::endl;

  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif

  // Create host/device memory
  int kernelsize, direction;
  float* h_kernel;
  InPixType *d_in;
  OutPixType *d_out;
  cudaMalloc((void**) &d_in,  sizeof(InPixType)*w*h*d);  CUT_CHECK_ERROR("Memory creation failed");
  cudaMalloc((void**) &d_out, sizeof(OutPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
 
  direction = oper.GetDirection();
  kernelsize = oper.GetSize(direction);
  assert(kernelsize <= MAX_KERNEL_SIZE);
  h_kernel = (float*)malloc(sizeof(float)*(kernelsize));
    
  // copy kernel weight
  for(int j=0; j<kernelsize; j++) h_kernel[j] = oper[j];   
   
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
#endif

  // copy host to device memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif
  cudaMemcpy(d_in, in->GetBufferPointer(), sizeof(InPixType)*w*h*d, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy failed");
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif


#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[0]));
#endif

  FILTERTYPE filterType;

  if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 1) filterType = UCHAR_TO_UCHAR;
  else if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 4) filterType = UCHAR_TO_FLOAT;
  else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
  else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 1) filterType = FLOAT_TO_UCHAR;
  else
  {
    printf("No such filter type\n");
    assert(false); exit(0);
  }

 
  cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                       (OutPixType*)d_out,
                                       h_kernel, filterType, 
                                       direction, direction, 
                                       kernelsize, w, h, d);



#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[0]));
#endif
  
  // copy device to host memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif

  cudaMemcpy(out->GetBufferPointer(), d_out, sizeof(OutPixType)*w*h*d, cudaMemcpyDeviceToHost);
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif

  // copy host to itkImage
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif

  // free dynamically allocated memory
  cudaFree((void*)d_in);  CUT_CHECK_ERROR("Memory free failed");
  cudaFree((void*)d_out); CUT_CHECK_ERROR("Memory free failed");
  free(h_kernel);  

#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
 
  printf("Kernel Time: %f (ms)\n", cutGetTimerValue(timer[0]));
  printf("CPU <-> CPU Time: %f (ms)\n", cutGetTimerValue(timer[1]));
  printf("CPU <-> GPU Time: %f (ms)\n", cutGetTimerValue(timer[2]));
  
  CUT_SAFE_CALL(cutDeleteTimer(timer[0]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[1]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[2]));
#endif

}

//
// Perona & Malik Anisotropic Diffusion
// 
template <class InputImageType, class OutputImageType>
void
cudaGradientAnisotropicDiffusionImageFilter(const InputImageType *in, OutputImageType *out, int numiter)
{
  typedef typename InputImageType::PixelType InPixType;
  typedef typename OutputImageType::PixelType OutPixType;
  
#ifdef DEBUG
  unsigned int timer[3];
  timer[0] = timer[1] = timer[2] = 0;
  CUT_SAFE_CALL(cutCreateTimer(&(timer[0])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[1])));
  CUT_SAFE_CALL(cutCreateTimer(&(timer[2])));
#endif

  int w, h, d, dim;
 
  typename InputImageType::SizeType imageSize = in->GetLargestPossibleRegion().GetSize();
  dim = in->GetImageDimension();

  // Dimension of the input image should be either 2 or 3
  if(dim < 1 || dim > 3)
  {
    std::cerr << "cudaMeanImageFilter only accepts 1/2/3D images" << std::endl;
    assert(false);
  } 

  w = imageSize[0];
  if(dim == 1) 
  {
    h = d = 1;
  }
  else if(dim == 2)
  { 
    h = imageSize[1];
    d = 1;
  }
  else
  {
    h = imageSize[1];
    d = imageSize[2];
  }

#ifdef DEBUG
  // debug : print out Gaussian kernel
  for(int i=0; i<dim; i++)
  {
    printf("Kernel for dim %d\n", i+1);
    double sum = 0;
    for(int j=0; j<oper[i].GetSize(i); j++)
    {
      double val = ((oper[i].GetBufferReference()))[j];
      sum += val;
      printf("%f ", val);
    }
    printf("\nSum of kernel : %f\n\n", sum);
  }

  std::cout << "Image Dimension : " << dim << std::endl;
  std::cout << "Image Size : " << w << "x" << h << "x" << d << std::endl;
  std::cout << "Size of pixel type : " << sizeof(InPixType) << std::endl;

  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif
  // Create host/device memory
  InPixType *d_in;
  OutPixType *d_out;
  cudaMalloc((void**) &d_in,  sizeof(InPixType)*w*h*d);  CUT_CHECK_ERROR("Memory creation failed");
  cudaMalloc((void**) &d_out, sizeof(OutPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
  
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
#endif

  // copy host to device memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif
  cudaMemcpy(d_in, in->GetBufferPointer(), sizeof(InPixType)*w*h*d, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy failed");
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif


#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[0]));
#endif

  FILTERTYPE filterType;

  if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 1) filterType = UCHAR_TO_UCHAR;
  else if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 4) filterType = UCHAR_TO_FLOAT;
  else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
  else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 1) filterType = FLOAT_TO_UCHAR;
  else
  {
    printf("No such filter type\n");
    assert(false); exit(0);
  }

  //assert(filterType == FLOAT_TO_FLOAT);

  cudaGradientAnisotropicDiffusionFilterKernelWrapper((InPixType*)d_in,
                                                      (OutPixType*)d_out,
                                                       filterType, dim, numiter,
                                                       anisoParam,
                                                       w, h, d);

#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[0]));
#endif
  
  // copy device to host memory
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[2]));
#endif

  cudaMemcpy(out->GetBufferPointer(), d_out, sizeof(OutPixType)*w*h*d, cudaMemcpyDeviceToHost);
#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[2]));
#endif

  // copy host to itkImage
#ifdef DEBUG
  CUT_SAFE_CALL(cutStartTimer(timer[1]));
#endif

  // free dynamically allocated memory
  cudaFree((void*)d_in);  CUT_CHECK_ERROR("Memory free failed");
  cudaFree((void*)d_out); CUT_CHECK_ERROR("Memory free failed");
  

#ifdef DEBUG
  CUT_SAFE_CALL(cutStopTimer(timer[1]));
 
  printf("Kernel Time: %f (ms)\n", cutGetTimerValue(timer[0]));
  printf("CPU <-> CPU Time: %f (ms)\n", cutGetTimerValue(timer[1]));
  printf("CPU <-> GPU Time: %f (ms)\n", cutGetTimerValue(timer[2]));
  
  CUT_SAFE_CALL(cutDeleteTimer(timer[0]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[1]));
  CUT_SAFE_CALL(cutDeleteTimer(timer[2]));
#endif

}

//
// compute Hessian of the image
//
template <class InputImageType, class OutputImageType>
void cudaHessianImageFilter(const InputImageType *_in, OutputImageType *_out, float sigma)
{
  typedef typename InputImageType::PixelType  InPixType;
  typedef typename OutputImageType::PixelType OutPixType;

  int w, h, d, dim;
 
  typename InputImageType::SizeType imageSize = _in->GetLargestPossibleRegion().GetSize();
  dim = _in->GetImageDimension();

  // Dimension of the input image should be either 2 or 3
  if(dim < 1 || dim > 3)
  {
    std::cerr << "cudaMeanImageFilter only accepts 1/2/3D images" << std::endl;
    assert(false);
  } 

  w = h = d = imageSize[0];
  if(dim == 2) h = imageSize[1];
  if(dim == 3)
  {
    h = imageSize[1];
    d = imageSize[2];
  }



  // create zero, first, and second Gaussian derivative filter
  //
  // Zero   : G = exp(-x^2/(2*sigma^2))
  // First  : G = x*exp(-x^2/(2*sigma^2))
  // Second : G = (1-(x/sigma)^2)*exp(-x^2/(2*sigma^2))
  //
  /*
  if (_in->GetSpacing()[i] > 0)
  {
    // convert the variance from physical units to pixels
    double s = _in->GetSpacing()[i];
    s = s*s;
    w_sigma = sigma/s;
  }   
  else w_sigma = sigma;
*/


  float w_sigma; // weight for sigma
  float *h_out;
  float* h_kernel[3];
  float* d_temp[2];
  float* d_in;

  w_sigma = sigma;

  int halfKernelWidth = ceil(sigma*5.0);
  int maxKernelWidth = halfKernelWidth*2 + 1;

  float kernelsum = 0;
  for(int i=0; i<3; i++) 
  {   
    h_kernel[i] = (float*)malloc(sizeof(float)*(maxKernelWidth));
  }

  for(int i=0; i<=halfKernelWidth; i++)
  {
    float val;
    float _x = i;

    // order zero
    val = 1.0f/exp((_x*_x)/(2.0f*w_sigma*w_sigma));
    kernelsum += val;
    if(i > 0) kernelsum += val;
    h_kernel[0][halfKernelWidth+i] = h_kernel[0][halfKernelWidth-i] = val;
  }

  for(int i=0; i<maxKernelWidth; i++)
  {
    float _x = i - halfKernelWidth;

    h_kernel[0][i] /= kernelsum;
    h_kernel[1][i] = (_x/(-(w_sigma*w_sigma)))*h_kernel[0][i];
    h_kernel[2][i] = ((1-(_x/w_sigma)*(_x/w_sigma))/(-(w_sigma*w_sigma)))*h_kernel[0][i];
  }
 
  //
  // Compute Derivatives
  // 

  // create output host memory
  cuMemAllocHost((void**)&h_out,sizeof(float)*w*h*d); // pinned memory
  //h_out = (float*)malloc(sizeof(float)*w*h*d);
  cudaMalloc((void**) &d_in, sizeof(InPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
  cudaMalloc((void**) &d_temp[0], sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
  cudaMalloc((void**) &d_temp[1], sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
  cudaMemcpy(d_in, _in->GetBufferPointer(), sizeof(InPixType)*w*h*d, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy failed");

  typedef NthElementImageAdaptor< OutputImageType, float >  OutputImageAdaptorType;
  typedef typename OutputImageAdaptorType::Pointer OutputImageAdaptorPointer;
  OutputImageAdaptorPointer imageAdaptor = OutputImageAdaptorType::New();

  imageAdaptor->SetImage( _out );
  imageAdaptor->SetLargestPossibleRegion( _in->GetLargestPossibleRegion() );
  imageAdaptor->SetBufferedRegion( _in->GetBufferedRegion() );
  imageAdaptor->SetRequestedRegion( _in->GetRequestedRegion() );
  imageAdaptor->Allocate();

  FILTERTYPE filterType;

  // compute Hessian
  int elem = 0;
  for(int i=0; i<dim; i++)
  {
    for(int j=i; j<dim; j++)
    {
      // initial filtertype depends on the input image      
      if(sizeof(InPixType) == 1) filterType = UCHAR_TO_FLOAT;
      else if(sizeof(InPixType) == 4) filterType = FLOAT_TO_FLOAT;
      else
      {
        printf("No such filter type\n");
        assert(false); exit(0);
      }

      int maxorder = 1;
      if(i == j) maxorder = 2;
      float *src, *tar, *tmp;
      src = d_temp[1];
      tar = d_temp[0];

      for(int k=0; k<dim; k++)
      {
        int order = 0;
        if(k == i || k == j) order = maxorder;

        if(k == 0)
        {
          cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in, tar, h_kernel[order],
                                                filterType, k, (k+2)%3, maxKernelWidth, 
                                                w, h, d);
        }
        else
        {
          cudaConvolutionFilter1DKernelWrapper(src, tar, h_kernel[order],
                                               filterType, k, (k+2)%3, maxKernelWidth, 
                                               w, h, d);
        }
        filterType = FLOAT_TO_FLOAT;
        tmp = src; src = tar; tar = tmp;
      }
  
      // copy result
      imageAdaptor->SelectNthElement( elem++ );
      ImageRegionIteratorWithIndex< OutputImageAdaptorType > ot( imageAdaptor, 
                                                                 imageAdaptor->GetRequestedRegion() );

      cudaMemcpy(h_out, src, sizeof(float)*w*h*d, cudaMemcpyDeviceToHost);
 
      unsigned int idx = 0;
      ot.GoToBegin();
      while( !ot.IsAtEnd() )
      {
        ot.Set( h_out[idx++] ); // type conversion
        ++ot;
      } 
    }
  }

  //free(h_out);
  cuMemFreeHost(h_out);
  cudaFree((void*)d_in); CUT_CHECK_ERROR("Memory free failed");
  cudaFree((void*)d_temp[0]); CUT_CHECK_ERROR("Memory free failed");
  cudaFree((void*)d_temp[1]); CUT_CHECK_ERROR("Memory free failed");


/*
  
  //
  // 1. Gaussian Smoothing
  //

  typedef GaussianOperator<float, ::itk::GetImageDimension<InputImageType>::ImageDimension> GaussianOperatorType;
  std::vector<GaussianOperatorType> gaussOper;
  gaussOper.resize(dim);

  // Set up the operators
  unsigned int maxKernelWidth = ceil(sigma*5.0)*2 + 1;
  for (int i = 0; i < dim; ++i)
  {
    // Set up the operator for this dimension
    gaussOper[i].SetDirection(i);

    if (_in->GetSpacing()[i] > 0)
    {
      // convert the variance from physical units to pixels
      double s = _in->GetSpacing()[i];
      s = s*s;
      gaussOper[i].SetVariance(sigma/s);
    }   
    else gaussOper[i].SetVariance(sigma);

    gaussOper[i].SetMaximumKernelWidth(maxKernelWidth);
    gaussOper[i].SetMaximumError(0.0001);
    gaussOper[i].CreateDirectional();
  }

  // Create host/device memory
  int kernel_size[3];
  float* h_kernel[3];
  float* d_temp[2];  
  InPixType  *d_in;
  float *h_out;
  float *d_inter; // intermediate result
 
  cudaMalloc((void**) &d_in, sizeof(InPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
  cudaMalloc((void**) &d_inter, sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
 
  // create Gaussian kernels
  for(int i=0; i<dim; i++)
  {
    kernel_size[i] = gaussOper[i].GetSize(i);
    assert(kernel_size[i] <= MAX_KERNEL_SIZE);

    h_kernel[i] = (float*)malloc(sizeof(float)*(kernel_size[i]));
   
    // copy Gaussian kernel weight
    for(int j=0; j<kernel_size[i]; j++)
    {
      h_kernel[i][j] = gaussOper[i][j];   
    }  
  }

  cudaMalloc((void**) &d_temp[0], sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");
  cudaMalloc((void**) &d_temp[1], sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");

  CUT_CHECK_ERROR("Memory creation failed");
 
  // copy input image from host to device memory
  cudaMemcpy(d_in, _in->GetBufferPointer(), sizeof(InPixType)*w*h*d, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy failed");

  FILTERTYPE filterType;

  if(dim == 1)
  {  
    // filtering along x
    if(sizeof(InPixType) == 1) filterType = UCHAR_TO_FLOAT;
    else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type\n");
      assert(false); exit(0);
    }
    cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                         (float*)d_inter,
                                         h_kernel[0],
                                         filterType, 0, 
                                         gaussOper[0].GetSize(0),
                                         w, h, d);

  }
  else if(dim == 2)
  { 
    // filtering along x
    if(sizeof(InPixType) == 1) filterType = UCHAR_TO_FLOAT;
    else if(sizeof(InPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type\n");
      assert(false); exit(0);
    }   
    cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                         (float*)d_temp[0],
                                          h_kernel[0],
                                          filterType, 0, 
                                          gaussOper[0].GetSize(0), w, h, d);

    // filtering along y
    filterType = FLOAT_TO_FLOAT;  
    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[0],
                                         (float*)d_inter,
                                          h_kernel[1],
                                          filterType, 1, 
                                          gaussOper[1].GetSize(1), w, h, d);                                                  
  }
  else if(dim == 3)
  { 
    // filtering along x
    if(sizeof(InPixType) == 1) filterType = UCHAR_TO_FLOAT;
    else if(sizeof(InPixType) == 4) filterType = FLOAT_TO_FLOAT;
    else
    {
      printf("No such filter type\n");
      assert(false); exit(0);
    }   
    cudaConvolutionFilter1DKernelWrapper((InPixType*)d_in,
                                         (float*)d_temp[0],
                                          h_kernel[0],
                                          filterType, 0, 
                                          gaussOper[0].GetSize(0), w, h, d);
    // filtering along y
    filterType = FLOAT_TO_FLOAT;    
    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[0],
                                         (float*)d_temp[1],
                                          h_kernel[1],
                                          filterType, 1, 
                                          gaussOper[1].GetSize(1), w, h, d);
   
    cudaConvolutionFilter1DKernelWrapper((float*)d_temp[1],
                                         (float*)d_inter,
                                          h_kernel[2],
                                          filterType, 2, 
                                          gaussOper[2].GetSize(2), w, h, d);
  }
  else
  {
     printf("Only 1/2/3D input image is supported!\n");
     assert(false); exit(0);
  }

  // free dynamically allocated memory
  cudaFree((void*)d_in);  CUT_CHECK_ERROR("Memory free failed");
  for(int i=0; i<dim; i++)
  {
    free(h_kernel[i]);    
  }


  //
  // 2. Compute Derivatives
  // 

  // create output host memory
  h_out = (float*)malloc(sizeof(float)*w*h*d);
  
  float h_gradKernel[3] = {-0.5,0,0.5};
  float h_currentGradKernel[3];

  filterType = FLOAT_TO_FLOAT; 

  typedef NthElementImageAdaptor< OutputImageType, float >  OutputImageAdaptorType;
  typedef typename OutputImageAdaptorType::Pointer OutputImageAdaptorPointer;
  OutputImageAdaptorPointer imageAdaptor = OutputImageAdaptorType::New();

  imageAdaptor->SetImage( _out );
  imageAdaptor->SetLargestPossibleRegion( _in->GetLargestPossibleRegion() );
  imageAdaptor->SetBufferedRegion( _in->GetBufferedRegion() );
  imageAdaptor->SetRequestedRegion( _in->GetRequestedRegion() );
  imageAdaptor->Allocate();

  // compute Hessian
  int elem = 0;
  for(int i=0; i<dim; i++)
  {
    for(int j=i; j<dim; j++)
    {
      for(int k=0; k<3; k++) h_currentGradKernel[k] = h_gradKernel[k]/_in->GetSpacing()[i];
    
      cudaConvolutionFilter1DKernelWrapper((float*)d_inter,
                                           (float*)d_temp[0],
                                           h_currentGradKernel,
                                           filterType, i, 3, 
                                           w, h, d);

      for(int k=0; k<3; k++) h_currentGradKernel[k] = h_gradKernel[k]/_in->GetSpacing()[j];

      cudaConvolutionFilter1DKernelWrapper((float*)d_temp[0],
                                           (float*)d_temp[1],
                                           h_currentGradKernel,
                                           filterType, j, 3, 
                                           w, h, d);
  
      // copy result
      imageAdaptor->SelectNthElement( elem++ );
      ImageRegionIteratorWithIndex< OutputImageAdaptorType > ot( imageAdaptor, 
                                                                 imageAdaptor->GetRequestedRegion() );

      //cudaMemcpy(imageAdaptor->GetBufferPointer(), d_temp[1], sizeof(float)*w*h*d, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_out, d_temp[1], sizeof(float)*w*h*d, cudaMemcpyDeviceToHost);
 
      unsigned int idx = 0;
      ot.GoToBegin();
      while( !ot.IsAtEnd() )
      {
        ot.Set( h_out[idx++] ); // type conversion
        ++ot;
      } 
    }
  }

  free(h_out);
  cudaFree((void*)d_inter); CUT_CHECK_ERROR("Memory free failed");
  cudaFree((void*)d_temp[0]); CUT_CHECK_ERROR("Memory free failed");
  cudaFree((void*)d_temp[1]); CUT_CHECK_ERROR("Memory free failed");
*/

}

template <class InputImageType, class OutputImageType, class InputSizeType>
void cudaMedianImageFilter(const InputImageType *_in, OutputImageType *_out, InputSizeType _radius)
{
  typedef typename InputImageType::SizeType sizetype;
  int w, h, d, dim;

  sizetype imageSize = _in->GetLargestPossibleRegion().GetSize();
  dim = _in->GetImageDimension();
  w = imageSize[0];
  h = imageSize[1];
  d = imageSize[2];

  // Dimension of the input image should be either 2 or 3
  if(dim < 2 || dim > 3)
  {
    std::cerr << "cudaMeanImageFilter only accepts 2/3D images" << std::endl;
    assert(false);
  } 

  // Setting block size
  if(dim == 2) d = 1;

  std::cout << "Image Dimension : " << dim << std::endl;
  std::cout << "Image Size : " << w << "x" << h << "x" << d << std::endl;
  std::cout << "Size of pixel type : " << sizeof(typename InputImageType::PixelType) << std::endl;
  
  // Median filter only supports 8-bit pixel type
  if(sizeof(typename InputImageType::PixelType) != 1)
  {
    std::cerr << "ERROR : Current CUDA MedianImageFilter only supports 8-bit pixel type" << std::endl;
    assert(false);
    exit(0);
  }

  // Create host/device memory
  typename InputImageType::PixelType *d_mem[2];  
  cudaMalloc((void**) &d_mem[0], sizeof(typename InputImageType::PixelType)*w*h*d);
  cudaMalloc((void**) &d_mem[1], sizeof(typename OutputImageType::PixelType)*w*h*d);
  CUT_CHECK_ERROR("Memory creation failed");
 
  // copy host to device memory
  cudaMemcpy(d_mem[0], _in->GetBufferPointer(), sizeof(typename InputImageType::PixelType)*w*h*d, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy failed");

  // call filter function
  int kw, kh, kd; // Mean filter kernel width/height/depth
  kw = kh = kd = 0;
  if(dim > 0) kw = _radius[0];
  if(dim > 1) kh = _radius[1];
  if(dim > 2) kd = _radius[2];  

  // type check -- should be better than this but will be fixed later
  if(sizeof(typename InputImageType::PixelType)==1)
  {
    cudaMedianImageFilterKernelWrapper((typename InputImageType::PixelType*)d_mem[0], 
                                       (typename OutputImageType::PixelType*)d_mem[1], 
                                       UCHAR_TO_UCHAR, dim, w, h, d, kw, kh, kd); 
  }
  else
  {
    assert(false);
    exit(0);
  }
  
  // copy device to host memory
  cudaMemcpy(_out->GetBufferPointer(), d_mem[1], sizeof(typename OutputImageType::PixelType)*w*h*d, cudaMemcpyDeviceToHost);
  CUT_CHECK_ERROR("Memory copy failed");

  // Free device memory
  cudaFree(d_mem[0]);
  cudaFree(d_mem[1]);
}
   

} // namespace

#endif
