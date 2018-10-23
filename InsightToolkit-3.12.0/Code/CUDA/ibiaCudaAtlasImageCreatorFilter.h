#ifndef __ibiaCudaAtlasImageCreatorFilter_h
#define __ibiaCudaAtlasImageCreatorFilter_h

#include "itkImage.h"
#include "itkCudaCommon.h"

/**
\class ibiaCudaAtlasImageCreatorFilter
\brief Wrapper for the CudaAtlasImageCreator Filter. It holds the device pointers for the input pictures persistently, so they don't have to be copied for every single kernel call.

  @author Thomas Wieland <thomas.wieland@umit.at>
  @version 1.0
*/

extern "C"
void cudaAtlasFilterKernelWrapper(void *d_in, void *d_out, void *d_principalComponents, void *d_atlasDist,
                                          FILTERTYPE type, int numberOfPrincipalComponents, 
					                      float atlasDefaultPixelValue, float *rootEvTimesSd, 
                                          uint image_width, uint image_height, uint image_depth, uint vectorFactor);

namespace ibia
{

template <class TInputImage, class TOutputImage, class ImageContainerDouble>
class ITK_EXPORT CudaAtlasImageCreatorFilter 

  {

public:
 
  /** Needed Types */
  
  typedef typename TInputImage::PixelType  InPixType;
  typedef typename TOutputImage::PixelType OutPixType;
   
  typedef typename TOutputImage::Pointer	OutPointerType;
  typedef typename TInputImage::Pointer		InPointerType;



  /** Methods */

  /** Standard constructor. **/
  CudaAtlasImageCreatorFilter()
{
  numberOfPrincipalComponents=w=h=d=dim=0;
  d_in=d_out=d_principalComponents= NULL;
  d_atlasDist = NULL;
  aktInput=NULL;
}

  /** Standard destructor. **/
  ~CudaAtlasImageCreatorFilter()
  { 
  	FreeMemory();
  }

  /** Set atlas distance map. **/
  template <class TAtlasImage>
  void SetAtlasDist(TAtlasImage *atlasDist)
  {
   	if(dim == 0)
		SetDimension<TAtlasImage>(atlasDist);
    
    	if(d_atlasDist == NULL)
	{
		cudaMalloc((void**) &d_atlasDist, sizeof(float)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");	//Allocate device memory if necessary
	}
	cudaMemcpy(d_atlasDist, atlasDist->GetBufferPointer(), sizeof(float)*w*h*d, cudaMemcpyHostToDevice);		//Copy data from host to device
  	CUT_CHECK_ERROR("Memory copy failed");  
  }


  /** Set input image **/
  void SetInputImage(InPointerType in){
  	
    if(dim == 0)
		SetDimension<TInputImage>(in);

	if(d_in == NULL)
	{
		cudaMalloc((void**) &d_in, sizeof(InPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");   //Allocate device memory if necessary
	}
	if(in != aktInput){
		aktInput = in;
		cudaMemcpy(d_in, in->GetBufferPointer(), sizeof(InPixType)*w*h*d, cudaMemcpyHostToDevice);	//Copy data from host to device
  		CUT_CHECK_ERROR("Memory copy failed");
	}

  }

  /** Set principal Components **/
  void SetPrincipalComponents(ImageContainerDouble principalComponents, int numberOfComponents)
  {
	if(d_principalComponents != NULL && numberOfPrincipalComponents != numberOfComponents)
	{
		cudaFree(d_principalComponents);
		CUT_CHECK_ERROR("Memory free failed");
		d_principalComponents = NULL;
	}
	numberOfPrincipalComponents = numberOfComponents;

	if(dim == 0)
		SetDimension<TInputImage>(principalComponents[0]);
	
	if(d_principalComponents == NULL)
	{
		cudaMalloc((void**) &d_principalComponents,  w*h*d*sizeof(InPixType)*numberOfPrincipalComponents); //Allocate device memory if necessary
		CUT_CHECK_ERROR("Memory creation failed");
	}

	for(int i = 0; i < numberOfPrincipalComponents; i++)
  	{
  		cudaMemcpy((void *)&(d_principalComponents[(i * w*h*d)]), principalComponents[i]->GetBufferPointer(), w*h*d*sizeof(InPixType), cudaMemcpyHostToDevice);	//Copy data from host to device
  		CUT_CHECK_ERROR("Memory copy failed");
  	}
  }
  
  /** Run the kernel */
  void RunKernel(OutPointerType out, float *rootEvTimesSd, float atlasDefaultPixelValue)
  {
	  cudaMalloc((void**) &d_out, sizeof(OutPixType)*w*h*d); CUT_CHECK_ERROR("Memory creation failed");  //Allocate device memory if necessary
	  
	  uint vectorFactor = 1;	
	  	//Determining FilterType
	  FILTERTYPE filterType;
	  if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 1) filterType = UCHAR_TO_UCHAR;
	  else if(sizeof(InPixType) == 1 && sizeof(OutPixType) == 4) filterType = UCHAR_TO_FLOAT;
	  else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 4) filterType = FLOAT_TO_FLOAT;
	  else if(sizeof(InPixType) == 4 && sizeof(OutPixType) == 1) filterType = FLOAT_TO_UCHAR;
	  else if(sizeof(InPixType) == dim   && sizeof(OutPixType) == dim){
	  	   filterType = UCHAR_TO_UCHAR;
	  	   vectorFactor = dim;
	  }
	  else if(sizeof(InPixType) == dim   && sizeof(OutPixType) == 4*dim){
	  	   filterType = UCHAR_TO_FLOAT;
	  	   vectorFactor = dim;
	  }
	  else if(sizeof(InPixType) == 4*dim && sizeof(OutPixType) == 4*dim){
	  	   filterType = FLOAT_TO_FLOAT;
	  	   vectorFactor = dim;
	  }
	  else if(sizeof(InPixType) == 4*dim && sizeof(OutPixType) == dim){
	  	   filterType = FLOAT_TO_UCHAR;
	  	   vectorFactor = dim;
	  }
	  else
	  {
	    printf("No such filter type\n");
	    assert(false); exit(0);
	  }
	  
	  cudaAtlasFilterKernelWrapper(d_in, d_out, d_principalComponents, d_atlasDist,		//Calling Kernelwrapper
                               filterType, numberOfPrincipalComponents, atlasDefaultPixelValue, 
                               rootEvTimesSd, w*vectorFactor, h, d, vectorFactor);
  
  
  
  	  // copy device to host memory
  	  cudaMemcpy(out->GetBufferPointer(), d_out, sizeof(OutPixType)*w*h*d, cudaMemcpyDeviceToHost);
	  CUT_CHECK_ERROR("Memory copy failed");
	  
	  cudaFree(d_out);
	  CUT_CHECK_ERROR("Memory free failed");
  
  }
  
  /** Free memory**/
  void FreeMemory()
  {
  	if(d_in != NULL)
  		cudaFree(d_in);
  	if(d_atlasDist != NULL)
  		cudaFree(d_atlasDist);
  	if(d_principalComponents != NULL)
  		cudaFree(d_principalComponents);
  	CUT_CHECK_ERROR("Memory free failed");
  }

  /** Set image dimension */
  template <class ImageType>
  void SetDimension(ImageType *in){
   	typename ImageType::SizeType imageSize = in->GetLargestPossibleRegion().GetSize();
   
    	dim = in->GetImageDimension();

   
   	if(dim < 1 || dim > 3)
   	{
   		std::cerr << "cudaAtlasImageCreatorFilter only accepts 1/2/3D images" << std::endl;
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
  }
  
  private:
    /** Needed variables */
  InPixType *d_in, *d_principalComponents;
  OutPixType *d_out;
  float *d_atlasDist;
  int w, h, d, dim, numberOfPrincipalComponents;
  InPointerType aktInput;

  };
}
#endif
