#ifndef __ibiaAtlasImageCreatorFilter_h
#define __ibiaAtlasImageCreatorFilter_h

#include "itkImageToImageFilter.h"
#include "itkArray.h"
#include "itkImageRegionIterator.h"
#include "itkRealTimeClock.h"
#include "ibiaCudaAtlasImageCreatorFilter.h"

// tw--
#include "itkCuda.h"

namespace ibia
{

/** \class AtlasImageCreatorFilter
 * \brief Calculates the atlas image according to set model parameters.
 *
 * This filter calculates the atlas image w.r.t. the current standard
 * deviations (model parameters) for following warping. It is implemented as
 * multithreaded filter in order to accelerate calculation.
 *
 * UPDATE: Supports itk::Vector as PixelType
 * CUDA Version: This version of the filter supports GPU computing on nVidia cards.
 * Set environment variable ITK_CUDA=1 to use it, and ITK_CUDA_TIME=1 to activate timers.
 *
 * NOTE: The input is expected to be the mean image.
 *
 * @author Philipp Steininger <philipp.steininger@umit.at>, Thomas Wieland <thomas.wieland@umit.at>
 * @version 2.0
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT AtlasImageCreatorFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef AtlasImageCreatorFilter Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(AtlasImageCreatorFilter, itk::ImageToImageFilter);

  /** Typedef to describe the output image region type. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  typedef typename TOutputImage::RegionType::SizeType OutputImageSizeType;
  typedef typename TOutputImage::RegionType::IndexType OutputImageIndexType;

  /** Inherit some types from the superclass. */
  typedef typename Superclass::OutputImageType OutputImageType;
  typedef typename Superclass::OutputImagePointer OutputImagePointer;
  typedef typename Superclass::InputImageType InputImageType;
  typedef typename Superclass::InputImagePointer InputImagePointer;
  typedef typename Superclass::OutputImageType::PixelType OutputPixelType;

  /** needed types **/
  typedef double ParametersValueType;
  typedef itk::Array<ParametersValueType> ParametersType;
  typedef itk::Image<float, InputImageType::ImageDimension> AtlasImageType;

  typedef std::vector<typename InputImageType::Pointer>  ImageContainerDouble;


  /** This method is used to set the state of the filter before
    * multi-threading. */
  virtual void BeforeThreadedGenerateData();

  /** This method is used to set the state of the filter after
    * multi-threading. */
  virtual void AfterThreadedGenerateData();

  /** Set/get number of principal components (model parameters). **/
  itkSetMacro(NumberOfPrincipalComponents, unsigned int);
  itkGetMacro(NumberOfPrincipalComponents, unsigned int);
  

  /** Set standard deviations for model. **/
  void SetStandardDeviations(const ParametersType &parameters)
  {
    m_StandardDeviations = parameters;
  }
  /** Get standard deviations for model. **/
  const ParametersType &GetStandardDeviations(void)
  {
    return m_StandardDeviations;
  }

  /** Set model eigenvalues. **/
  void SetEigenvalues(const ParametersType &parameters)
  {
    m_Eigenvalues = parameters;
  }
  /** Get model eigenvalues. **/
  const ParametersType &GetEigenvalues(void)
  {
    return m_Eigenvalues;
  }

  /** Set principal component field container. **/
  void SetPrincipalComponentImageContainer(const
    ImageContainerDouble &newContainer)
  {
    m_PrincipalComponentImageContainer = newContainer;
    // tw- number of principal components must be set before calling this method
    if(itk::checkCUDA())
    	cudaFilter.SetPrincipalComponents(m_PrincipalComponentImageContainer, m_NumberOfPrincipalComponents);
  }
  /** Get principal component field container. **/
 ImageContainerDouble &GetPrincipalComponentImageContainer()
  {
    return m_PrincipalComponentImageContainer;
  }

  /** Set/get atlas distance map. **/
  void SetAtlasDist(AtlasImageType *atlasDist)
  {
    m_AtlasDist = atlasDist;
    if(itk::checkCUDA())
    	cudaFilter.SetAtlasDist(atlasDist);
  }
  AtlasImageType *GetAtlasDist()
  {
    return m_AtlasDist;
  }

  /** Set/get atlas image default pixel value **/
  itkSetMacro(AtlasDefaultPixelValue, float);
  itkGetMacro(AtlasDefaultPixelValue, float);

protected:
  /** needed types **/
  typedef itk::ImageRegionIterator<InputImageType> ImageIteratorType;
  typedef itk::ImageRegionIterator<AtlasImageType> AtlasIteratorType;

  /** number of principal components (model parameters) **/
  unsigned int m_NumberOfPrincipalComponents;
  /** array: sqrt(eigenvalue[])*standardDeviation[] **/
  float *m_RootEvTimesSd;
  /** model standard deviations **/
  ParametersType m_StandardDeviations;
  /** model eigenvalues **/
  ParametersType m_Eigenvalues;
  /** container for principal component deformation fields **/
 ImageContainerDouble m_PrincipalComponentImageContainer;
  /** atlas image default pixel value **/
  //OutputPixelType m_AtlasDefaultPixelValue;
  float m_AtlasDefaultPixelValue;
  /** atlas distance map **/
  typename AtlasImageType::Pointer m_AtlasDist;
  CudaAtlasImageCreatorFilter<InputImageType, OutputImageType, ImageContainerDouble> cudaFilter;	//tw-

  /** Standard constructor. **/
  AtlasImageCreatorFilter();
  /** Destructor. **/
  ~AtlasImageCreatorFilter();

  /** Print own information. **/
  void PrintSelf(std::ostream& os, itk::Indent indent) const;
  /** AtlasImageCreatorFilter is implemented as a multi-threaded filter.
    * As such, it needs to provide and implementation for
    * ThreadedGenerateData(). */
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
    int threadId);

private:
  AtlasImageCreatorFilter(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "ibiaAtlasImageCreatorFilter.txx"
#endif

#endif
