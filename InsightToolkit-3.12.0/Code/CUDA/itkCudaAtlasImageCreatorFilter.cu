#include "itkCudaCommon.h"


/**
\class ibiaCudaAtlasImageCreatorFilterKERNEL
\brief Cuda Kernel for the AtlasImageCreatorFilter

Needs no shared memory, because every pixel of the InputPictures is just read once.
The Outputpixel at position i calculated this way:
outputImage[i] = meanImage[] + { standardDeviations[j] * eigenvalues[j] * principalComponents[j][i] } _j=1..numberOfPrincipalComponents 

  @author Thomas Wieland <thomas.wieland@umit.at>
  @version 1.0
*/


#define MAX_NUMBER_OF_PRINCIPAL_COMPONENTS 128

/** constant memory that kernel can see */
__constant__ float d_rootEvTimesSd[MAX_NUMBER_OF_PRINCIPAL_COMPONENTS];

__global__ void kernel_AtlasFilter_F2F(float* d_in, float* d_out, float* d_principalComponents, float *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor);
__global__ void kernel_AtlasFilter_U2F(uchar* d_in, float* d_out, uchar* d_principalComponents, float *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor);
__global__ void kernel_AtlasFilter_F2U(float* d_in, uchar* d_out, float* d_principalComponents, uchar *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor);
__global__ void kernel_AtlasFilter_U2U(uchar* d_in, uchar* d_out, uchar* d_principalComponents, uchar *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor);

extern "C"
void cudaAtlasFilterKernelWrapper(void *d_in, void *d_out, void *d_principalComponents, void *d_atlasDist,
                                          FILTERTYPE type, int numberOfPrincipalComponents, float atlasDefaultPixelValue, float *rootEvTimesSd, 
                                          uint image_width, uint image_height, uint image_depth, uint vectorFactor)
{
  // 8 : max block width = 256
  // 7 : max block width = 128
  int maxBlockWidthPowSize = 7; // max block width = 2^maxBlockWidthPowSize

  uint blockWidth, blockHeight, blockDepth;
  uint nBlockX, nBlockY, nBlockZ;
   
  // copy rootEvTimesSd to constant memory
  cudaMemcpyToSymbol(d_rootEvTimesSd, rootEvTimesSd, sizeof(float)*numberOfPrincipalComponents);
  CUT_CHECK_ERROR("Copy to Symbol failed");

  // compute how many blocks are needed
  dim3 griddim; // actual grid dimension
  uint bestfitsize;
  
  bestfitsize = log2(image_height);
  blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
  blockHeight = 1;
  blockDepth  = 1;

  nBlockX = ceil((float)image_width / (float)blockWidth);
  nBlockY = image_height;
  nBlockZ = image_depth;
  griddim = dim3(nBlockX,nBlockY,nBlockZ);
  
  dim3 dimGrid(nBlockX*nBlockY,nBlockZ); // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight, blockDepth);
  dim3 imagedim(image_width,image_height,image_depth);
 
  CUT_CHECK_ERROR("BEFORE Kernel execution failed");
  // kernel call
  if(type == FLOAT_TO_FLOAT)
  {
    kernel_AtlasFilter_F2F<<<dimGrid,dimBlock>>>((float*)d_in, (float*)d_out, (float*) d_principalComponents, (float*) d_atlasDist, imagedim, griddim, numberOfPrincipalComponents, atlasDefaultPixelValue, vectorFactor);
  }
  else if(type == FLOAT_TO_UCHAR)
  {
    kernel_AtlasFilter_F2U<<<dimGrid,dimBlock>>>((float*)d_in, (uchar*)d_out, (float*) d_principalComponents, (uchar*) d_atlasDist, imagedim, griddim, numberOfPrincipalComponents, atlasDefaultPixelValue, vectorFactor);
  }
  else if(type == UCHAR_TO_FLOAT)
  {
    kernel_AtlasFilter_U2F<<<dimGrid,dimBlock>>>((uchar*)d_in, (float*)d_out, (uchar*) d_principalComponents, (float*) d_atlasDist, imagedim, griddim, numberOfPrincipalComponents, atlasDefaultPixelValue, vectorFactor);
  }
  else if(type == UCHAR_TO_UCHAR)
  {
    kernel_AtlasFilter_U2U<<<dimGrid,dimBlock>>>((uchar*)d_in, (uchar*)d_out, (uchar*) d_principalComponents, (uchar*) d_atlasDist, imagedim, griddim, numberOfPrincipalComponents, atlasDefaultPixelValue, vectorFactor);
  }
  else
  {
    printf("Unsupported pixel type\n");
    exit(0);
  }
  CUT_CHECK_ERROR("Kernel execution failed");
 
}

//////////////////////////////////////////////////
//						//
//		      KERNEL			//
//						//
//////////////////////////////////////////////////

__global__ void kernel_AtlasFilter_F2F(float* d_in, float* d_out, float* d_principalComponents, float *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor)
{
	
	//Check if the current pixel is within the picture
	if( ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x) < imagedim.x && ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y) < imagedim.y && (blockIdx.y * blockDim.z + threadIdx.z) < imagedim.z)
	{
		//Compute Index within memory: sliceoffset + lineoffset + current index
		uint currentIndex = (blockIdx.y * blockDim.z + threadIdx.z)*imagedim.x*imagedim.y + ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y)*imagedim.x + ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x);
		float outPixel = atlasDefaultPixelValue;
		
		//Check the current AtlasDist Value
		if( d_atlasDist[currentIndex/vectorFactor] >= 0.0)
		{
			outPixel = d_in[currentIndex];				//Compute the new pixel value
			uint imageSize = imagedim.x*imagedim.y*imagedim.z;
			
			for( int i = 0; i < numberOfPrincipalComponents; i++)
				outPixel += d_principalComponents[imageSize*i + currentIndex] *  d_rootEvTimesSd[i];

		}
		
		d_out[currentIndex] = outPixel;

	}
}


__global__ void kernel_AtlasFilter_U2F(uchar* d_in, float* d_out, uchar* d_principalComponents, float *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor)
{
		//Check if the current pixel is within the picture
	if( ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x) < imagedim.x && ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y) < imagedim.y && (blockIdx.y * blockDim.z + threadIdx.z) < imagedim.z)
	{
		//Compute Index within memory: sliceoffset + lineoffset + current index
		uint currentIndex = (blockIdx.y * blockDim.z + threadIdx.z)*imagedim.x*imagedim.y + ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y)*imagedim.x + ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x);
		float outPixel = atlasDefaultPixelValue;
		
		//Check the current AtlasDist Value
		if( d_atlasDist[currentIndex/vectorFactor] >= 0.0)
		{
			outPixel = d_in[currentIndex];				//Compute the new pixel value
			uint imageSize = imagedim.x*imagedim.y*imagedim.z;
			
			for( int i = 0; i < numberOfPrincipalComponents; i++)
				outPixel += d_principalComponents[imageSize*i + currentIndex] *  d_rootEvTimesSd[i];

		}
		
		d_out[currentIndex] = outPixel;

	}
	

	
}



__global__ void kernel_AtlasFilter_F2U(float* d_in, uchar* d_out, float* d_principalComponents, uchar *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor)
{
	//Check if the current pixel is within the picture
	if( ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x) < imagedim.x && ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y) < imagedim.y && (blockIdx.y * blockDim.z + threadIdx.z) < imagedim.z)
	{
		//Compute Index within memory: sliceoffset + lineoffset + current index
		uint currentIndex = (blockIdx.y * blockDim.z + threadIdx.z)*imagedim.x*imagedim.y + ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y)*imagedim.x + ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x);
		uchar outPixel = atlasDefaultPixelValue;
		
		//Check the current AtlasDist Value
		if( d_atlasDist[currentIndex/vectorFactor] >= 0.0)
		{
			outPixel = d_in[currentIndex];				//Compute the new pixel value
			uint imageSize = imagedim.x*imagedim.y*imagedim.z;
			
			for( int i = 0; i < numberOfPrincipalComponents; i++)
				outPixel += d_principalComponents[imageSize*i + currentIndex] *  d_rootEvTimesSd[i];

		}
		
		d_out[currentIndex] = outPixel;

	}
}




__global__ void kernel_AtlasFilter_U2U(uchar* d_in, uchar* d_out, uchar* d_principalComponents, uchar *d_atlasDist, dim3 imagedim, dim3 griddim, int numberOfPrincipalComponents, float atlasDefaultPixelValue, uint vectorFactor)
{
	//Check if the current pixel is within the picture
	if( ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x) < imagedim.x && ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y) < imagedim.y && (blockIdx.y * blockDim.z + threadIdx.z) < imagedim.z)
	{
		//Compute Index within memory: sliceoffset + lineoffset + current index
		uint currentIndex = (blockIdx.y * blockDim.z + threadIdx.z)*imagedim.x*imagedim.y + ((blockIdx.x / griddim.x) * blockDim.y + threadIdx.y)*imagedim.x + ((blockIdx.x % griddim.x) * blockDim.x + threadIdx.x);
		uchar outPixel = atlasDefaultPixelValue;
		
		//Check the current AtlasDist Value
		if( d_atlasDist[currentIndex/vectorFactor] >= 0.0)
		{
			outPixel = d_in[currentIndex];				//Compute the new pixel value
			uint imageSize = imagedim.x*imagedim.y*imagedim.z;
			
			for( int i = 0; i < numberOfPrincipalComponents; i++)
				outPixel += d_principalComponents[imageSize*i + currentIndex] *  d_rootEvTimesSd[i];

		}
		
		d_out[currentIndex] = outPixel;

	}
	
}




