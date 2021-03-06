/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkDiscreteGaussianImageFilter.txx,v $
Language:  C++
Date:      $Date: 2008-12-10 01:58:48 $
Version:   $Revision: 1.42 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkDiscreteGaussianImageFilter_txx
#define __itkDiscreteGaussianImageFilter_txx

#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkGaussianOperator.h"
#include "itkImageRegionIterator.h"
#include "itkProgressAccumulator.h"
#include "itkStreamingImageFilter.h"

// wk--
#include "itkCuda.h"

namespace itk
{
template <class TInputImage, class TOutputImage>
void 
DiscreteGaussianImageFilter<TInputImage,TOutputImage>
::GenerateInputRequestedRegion() throw(InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method. this should
  // copy the output requested region to the input requested region
  Superclass::GenerateInputRequestedRegion();
  
  // get pointers to the input and output
  typename Superclass::InputImagePointer  inputPtr = 
    const_cast< TInputImage *>( this->GetInput() );
  
  if ( !inputPtr )
    {
    return;
    }

  // Build an operator so that we can determine the kernel size
  GaussianOperator<OutputPixelType, ImageDimension> oper;
  typename TInputImage::SizeType radius;
  
  for (unsigned int i = 0; i < TInputImage::ImageDimension; i++)
    {
    // Determine the size of the operator in this dimension.  Note that the
    // Gaussian is built as a 1D operator in each of the specified directions.
    oper.SetDirection(i);
    if (m_UseImageSpacing == true)
      {
      if (this->GetInput()->GetSpacing()[i] == 0.0)
        {
        itkExceptionMacro(<< "Pixel spacing cannot be zero");
        }
      else
        {
        // convert the variance from physical units to pixels
        double s = this->GetInput()->GetSpacing()[i];
        s = s*s;
        oper.SetVariance(m_Variance[i] / s);
        }
      }
    else
      {
      oper.SetVariance(m_Variance[i]);
      }
    oper.SetMaximumError(m_MaximumError[i]);
    oper.SetMaximumKernelWidth(m_MaximumKernelWidth);
    oper.CreateDirectional();
    
    radius[i] = oper.GetRadius(i);
    }

  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion;
  inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius( radius );

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


template< class TInputImage, class TOutputImage >
void
DiscreteGaussianImageFilter<TInputImage, TOutputImage>
::GenerateData()
{

  unsigned int timer;
  if(checkTime())
  {
    // check running time
    timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&(timer)));
    CUT_SAFE_CALL(cutStartTimer(timer));
  }

  //printf("Size of pixel type : %d\n", sizeof(TInputImage::PixelType));

  typename TOutputImage::Pointer output = this->GetOutput();
  
  output->SetBufferedRegion(output->GetRequestedRegion());
  output->Allocate();

  // Create an internal image to protect the input image's metdata
  // (e.g. RequestedRegion). The StreamingImageFilter changes the
  // requested region as part of its normal processing.
  typename TInputImage::Pointer localInput = TInputImage::New();
  localInput->Graft(this->GetInput());

  // Determine the dimensionality to filter
  unsigned int filterDimensionality = m_FilterDimensionality;
  if (filterDimensionality > ImageDimension)
    {
    filterDimensionality = ImageDimension;
    }
  if (filterDimensionality == 0)
    {
    // no smoothing, copy input to output
    ImageRegionConstIterator<InputImageType> inIt(
      localInput,
      this->GetOutput()->GetRequestedRegion() );
    ImageRegionIterator<OutputImageType> outIt(
      output,
      this->GetOutput()->GetRequestedRegion() );

    while (!inIt.IsAtEnd())
      {
      outIt.Set( static_cast<OutputPixelType>(inIt.Get()) );
      ++inIt;
      ++outIt;
      }
    return;
    }
  
  // Type of the pixel to use for intermediate results
  typedef typename NumericTraits<OutputPixelType>::RealType RealOutputPixelType;
  typedef Image<OutputPixelType, ImageDimension> RealOutputImageType;
  
  // Type definition for the internal neighborhood filter
  //
  // First filter convolves and changes type from input type to real type
  // Middle filters convolves from real to real
  // Last filter convolves and changes type from real type to output type
  // Streaming filter forces the mini-pipeline to run in chunks
  typedef NeighborhoodOperatorImageFilter<InputImageType,
    RealOutputImageType, RealOutputPixelType> FirstFilterType;
  typedef NeighborhoodOperatorImageFilter<RealOutputImageType,
    RealOutputImageType, RealOutputPixelType> IntermediateFilterType;
  typedef NeighborhoodOperatorImageFilter<RealOutputImageType,
    OutputImageType, RealOutputPixelType> LastFilterType;
  typedef NeighborhoodOperatorImageFilter<InputImageType,
    OutputImageType, RealOutputPixelType> SingleFilterType;
  typedef StreamingImageFilter<OutputImageType, OutputImageType>
    StreamingFilterType;
  
  typedef typename FirstFilterType::Pointer FirstFilterPointer;
  typedef typename IntermediateFilterType::Pointer IntermediateFilterPointer;
  typedef typename LastFilterType::Pointer         LastFilterPointer;
  typedef typename SingleFilterType::Pointer       SingleFilterPointer;
  typedef typename StreamingFilterType::Pointer    StreamingFilterPointer;

  // Create a series of operators
  typedef GaussianOperator<RealOutputPixelType, ImageDimension> OperatorType;
  std::vector<OperatorType> oper;
  oper.resize(filterDimensionality);


  // Create a process accumulator for tracking the progress of minipipeline
  ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
  progress->SetMiniPipelineFilter(this);

  // Set up the operators
  unsigned int i;
  for (i = 0; i < filterDimensionality; ++i)
    {
    // Set up the operator for this dimension
    oper[i].SetDirection(i);
    if (m_UseImageSpacing == true)
      {
      if (localInput->GetSpacing()[i] == 0.0)
        {
        itkExceptionMacro(<< "Pixel spacing cannot be zero");
        }
      else
        {
        // convert the variance from physical units to pixels
        double s = localInput->GetSpacing()[i];
        s = s*s;
        oper[i].SetVariance(m_Variance[i] / s);
        }
      }
    else
      {
      oper[i].SetVariance(m_Variance[i]);
      }

    oper[i].SetMaximumKernelWidth(m_MaximumKernelWidth);
    oper[i].SetMaximumError(m_MaximumError[i]);
    oper[i].CreateDirectional();
    }

  // Create a chain of filters
  //
  //

  // wk-- CUDA
  if( checkCUDA() )
  {
    cudaDiscreteGaussianImageFilter<TInputImage, TOutputImage, OperatorType>(this->GetInput(), output, oper);
/*
    unsigned long int filterwidth = oper[0].GetSize(0);
    for(unsigned long int i=0; i<filterwidth; i++)
    {
      printf("%f ", oper[0][i]);
    }
    //cudaMeanImageFilter<InputImageType, OutputImageType, InputSizeType>(input, output, m_Radius);  
    */
     this->GraftOutput(output); // wk-- required??
  }
  else
  {
    if (filterDimensionality == 1)
      {
      // Use just a single filter
      SingleFilterPointer singleFilter = SingleFilterType::New();
      singleFilter->SetOperator(oper[0]);
      singleFilter->SetInput( localInput );
      progress->RegisterInternalFilter(singleFilter,1.0f/m_FilterDimensionality);

      // Graft this filters output onto the mini-pipeline so the mini-pipeline
      // has the correct region ivars and will write to this filters bulk data
      // output.
      singleFilter->GraftOutput( output );
      
      // Update the filter 
      singleFilter->Update();
      
      // Graft the last output of the mini-pipeline onto this filters output so
      // the final output has the correct region ivars and a handle to the final
      // bulk data
      this->GraftOutput(output);
      }
    else
      {
      // Setup a full mini-pipeline and stream the data through the
      // pipeline.
      unsigned int numberOfDivisions = ImageDimension*ImageDimension;
      unsigned int numberOfStages = filterDimensionality*numberOfDivisions + 1;
      
      // First filter convolves and changes type from input type to real type
      FirstFilterPointer firstFilter = FirstFilterType::New();
      firstFilter->SetOperator(oper[0]);
      firstFilter->ReleaseDataFlagOn();
      firstFilter->SetInput( localInput );
      progress->RegisterInternalFilter(firstFilter,1.0f/numberOfStages);
      
      // Middle filters convolves from real to real 
      std::vector<IntermediateFilterPointer> intermediateFilters;
      if (filterDimensionality > 2)
        {
        for (i=1; i < filterDimensionality-1; ++i)
          {
          IntermediateFilterPointer f = IntermediateFilterType::New();
          f->SetOperator(oper[i]);
          f->ReleaseDataFlagOn();
          progress->RegisterInternalFilter(f,1.0f / numberOfStages);
          
          if (i==1)
            {
            f->SetInput(firstFilter->GetOutput());
            }
          else
            {
            // note: first filter in vector (zeroth element) is for i==1
            f->SetInput(intermediateFilters[i-2]->GetOutput());
            }
          
          intermediateFilters.push_back( f );
          }
        }
      
      // Last filter convolves and changes type from real type to output type
      LastFilterPointer lastFilter = LastFilterType::New();
      lastFilter->SetOperator(oper[filterDimensionality-1]);
      lastFilter->ReleaseDataFlagOn();
      if (filterDimensionality > 2)
        {
        lastFilter->SetInput( intermediateFilters[filterDimensionality-3]->GetOutput() );
        }
      else
        {
        lastFilter->SetInput( firstFilter->GetOutput() );
        }
      progress->RegisterInternalFilter(lastFilter,1.0f / numberOfStages);
      
      // Put in a StreamingImageFilter so the mini-pipeline is processed
      // in chunks to minimize memory usage
      StreamingFilterPointer streamingFilter = StreamingFilterType::New();
      streamingFilter->SetInput( lastFilter->GetOutput() );
      streamingFilter->SetNumberOfStreamDivisions( numberOfDivisions );
      progress->RegisterInternalFilter(streamingFilter,1.0f / numberOfStages);
      
      // Graft this filters output onto the mini-pipeline so the mini-pipeline
      // has the correct region ivars and will write to this filters bulk data
      // output.
      streamingFilter->GraftOutput( output );
      
      // Update the last filter in the chain
      streamingFilter->Update();
      
      // Graft the last output of the mini-pipeline onto this filters output so
      // the final output has the correct region ivars and a handle to the final
      // bulk data
      this->GraftOutput(output);
      }
  }

  if(checkTime())
  {
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf("Total Time: %f (ms)\n", cutGetTimerValue(timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));
  }
}

template< class TInputImage, class TOutputImage >
void
DiscreteGaussianImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Variance: " << m_Variance << std::endl;
  os << indent << "MaximumError: " << m_MaximumError << std::endl;
  os << indent << "MaximumKernelWidth: " << m_MaximumKernelWidth << std::endl;
  os << indent << "FilterDimensionality: " << m_FilterDimensionality << std::endl;
  os << indent << "UseImageSpacing: " << m_UseImageSpacing << std::endl;
}

} // end namespace itk

#endif
