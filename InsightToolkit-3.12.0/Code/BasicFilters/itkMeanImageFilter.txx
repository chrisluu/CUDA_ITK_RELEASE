/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMeanImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2008-10-16 18:05:25 $
  Version:   $Revision: 1.16 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMeanImageFilter_txx
#define __itkMeanImageFilter_txx


// First make sure that the configuration is available.
// This line can be removed once the optimized versions
// gets integrated into the main directories.
#include "itkConfigure.h"

#ifdef ITK_USE_CONSOLIDATED_MORPHOLOGY
#include "itkOptMeanImageFilter.txx"
#else

#include "itkMeanImageFilter.h"

#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkOffset.h"
#include "itkProgressReporter.h"

namespace itk
{

template <class TInputImage, class TOutputImage>
MeanImageFilter<TInputImage, TOutputImage>
::MeanImageFilter()
{
  m_Radius.Fill(1);
  if(checkCUDA()) initCUDA(); // wk--
}

template <class TInputImage, class TOutputImage>
void 
MeanImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion() throw (InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();
  
  // get pointers to the input and output
  typename Superclass::InputImagePointer inputPtr = 
    const_cast< TInputImage * >( this->GetInput() );
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
  
  if ( !inputPtr || !outputPtr )
    {
    return;
    }

  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion;
  inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius( m_Radius );

  // crop the input requested region at the input's largest possible region
  if ( inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion()) )
    {
    inputPtr->SetRequestedRegion( inputRequestedRegion );
    return;
    }
  else
    {
    // Couldn't crop the region (requested region is outside the largest
    // possible region).  Throw an exception.

    // store what we tried to request (prior to trying to crop)
    inputPtr->SetRequestedRegion( inputRequestedRegion );
    
    // build an exception
    InvalidRequestedRegionError e(__FILE__, __LINE__);
    e.SetLocation(ITK_LOCATION);
    e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
    e.SetDataObject(inputPtr);
    throw e;
    }
}


template< class TInputImage, class TOutputImage>
void
MeanImageFilter< TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       int threadId)
{
  unsigned int i;
  ZeroFluxNeumannBoundaryCondition<InputImageType> nbc;

  ConstNeighborhoodIterator<InputImageType> bit;
  ImageRegionIterator<OutputImageType> it;
  
  // Allocate output
  typename OutputImageType::Pointer output = this->GetOutput();
  typename InputImageType::ConstPointer input  = this->GetInput();

  unsigned int timer;

  if(checkTime())
  {
    // check running time
    timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&(timer)));
    CUT_SAFE_CALL(cutStartTimer(timer));
  }

  // wk-- CUDA
  if( checkCUDA() )
  {
    cudaMeanImageFilter<InputImageType, OutputImageType, InputSizeType>(input, output, m_Radius);   
  }
  else
  {
    // Find the data-set boundary "faces"
    typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType faceList;
    NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType> bC;
    faceList = bC(input, outputRegionForThread, m_Radius);

    typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<InputImageType>::FaceListType::iterator fit;

    // support progress methods/callbacks
    ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());
    
    InputRealType sum;

    // Process each of the boundary faces.  These are N-d regions which border
    // the edge of the buffer.
    for (fit=faceList.begin(); fit != faceList.end(); ++fit)
    { 
      bit = ConstNeighborhoodIterator<InputImageType>(m_Radius,
                                                      input, *fit);
      unsigned int neighborhoodSize = bit.Size();
      it = ImageRegionIterator<OutputImageType>(output, *fit);
      bit.OverrideBoundaryCondition(&nbc);
      bit.GoToBegin();

      while ( ! bit.IsAtEnd() )
      {
        sum = NumericTraits<InputRealType>::Zero;
        for (i = 0; i < neighborhoodSize; ++i)
          {            
            sum += static_cast<InputRealType>( bit.GetPixel(i) );
          }
        
        // get the mean value -- wk : actual mean kernel
        it.Set( static_cast<OutputPixelType>(sum / double(neighborhoodSize)) );
        
        ++bit;
        ++it;
        progress.CompletedPixel();
      }
    }
  }


  if(checkTime())
  {
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf("Total Time: %f (ms)\n", cutGetTimerValue(timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
  }
}

/**
 * Standard "PrintSelf" method
 */
template <class TInputImage, class TOutput>
void
MeanImageFilter<TInputImage, TOutput>
::PrintSelf(
  std::ostream& os, 
  Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Radius: " << m_Radius << std::endl;

}

} // end namespace itk

#endif

#endif
