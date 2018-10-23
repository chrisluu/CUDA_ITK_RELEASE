/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: DerivativeImageFilter.cxx,v $
Language:  C++
Date:      $Date: 2005/08/31 13:55:21 $
Version:   $Revision: 1.23 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

//  Software Guide : BeginCommandLineArgs
//    INPUTS:  {BrainProtonDensitySlice.png}
//    OUTPUTS: {DerivativeImageFilterFloatOutput.mhd}
//    OUTPUTS: {DerivativeImageFilterOutput.png}
//    1 0
//  Software Guide : EndCommandLineArgs

//  Software Guide : BeginLatex
//
//  The \doxygen{DerivativeImageFilter} is used for computing the partial
//  derivative of an image, the derivative of an image along a particular axial
//  direction.
//
//  \index{itk::DerivativeImageFilter}
//
//  Software Guide : EndLatex 

#include <time.h>
#include <stdlib.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"

typedef float pixeltype; // unsigned char
typedef itk::Image<pixeltype, 3> ItkImageType;
typedef ItkImageType::Pointer ItkImagePointer;

//  Software Guide : BeginLatex
//
//  The header file corresponding to this filter should be included first.
//
//  \index{itk::DerivativeImageFilter!header}
//
//  Software Guide : EndLatex 


// Software Guide : BeginCodeSnippet
#include "itkDerivativeImageFilter.h"
// Software Guide : EndCodeSnippet

//
// MODE_SELFTEST : run filter on various size datasets and calculate timing
// MODE_INPUT    : run filters on the input image/volume
//
#define  MODE_SELFTEST //MODE_INPUT //


using namespace itk;

int main( int argc, char * argv[] )
{

	putenv("ITK_CUDA_TIME=1");

#ifdef MODE_INPUT

	/*
	if( argc < 6 )
	{
	std::cerr << "Usage: " << std::endl;
	std::cerr << argv[0] << "  inputImageFile   outputImageFile  normalizedOutputImageFile ";
	std::cerr << " derivativeOrder direction" << std::endl;
	return EXIT_FAILURE;
	}
	*/

	putenv("ITK_CUDA_TIME=1");
	//putenv("ITK_CUDA=1");

	argv[1] = "C:/Work/proj/InsightToolkit-3.2.0/Examples/Data/BrainProtonDensitySlice.png";
	argv[2] = "DerivativeImageFilterFloatOutput.mhd";
	argv[3] = "DerivativeImageFilterOutput.png";
	argv[4] = "1";
	argv[5] = "1"; // direction


	//  Software Guide : BeginLatex
	//
	//  Next, the pixel types for the input and output images must be defined and, with
	//  them, the image types can be instantiated. Note that it is important to
	//  select a signed type for the image, since the values of the derivatives
	//  will be positive as well as negative.
	//
	//  Software Guide : EndLatex 

	// Software Guide : BeginCodeSnippet
	typedef   float  InputPixelType;
	typedef   float  OutputPixelType;

	const unsigned int Dimension = 2;

	typedef itk::Image< InputPixelType,  Dimension >   InputImageType;
	typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;
	// Software Guide : EndCodeSnippet


	typedef itk::ImageFileReader< InputImageType  >  ReaderType;
	typedef itk::ImageFileWriter< OutputImageType >  WriterType;

	ReaderType::Pointer reader = ReaderType::New();
	WriterType::Pointer writer = WriterType::New();

	reader->SetFileName( argv[1] );
	writer->SetFileName( argv[2] );

	//  Software Guide : BeginLatex
	//
	//  Using the image types, it is now possible to define the filter type
	//  and create the filter object. 
	//
	//  \index{itk::DerivativeImageFilter!instantiation}
	//  \index{itk::DerivativeImageFilter!New()}
	//  \index{itk::DerivativeImageFilter!Pointer}
	// 
	//  Software Guide : EndLatex 

	// Software Guide : BeginCodeSnippet
	typedef itk::DerivativeImageFilter<
		InputImageType, OutputImageType >  FilterType;

	FilterType::Pointer filter[2];
	filter[0] = FilterType::New();
	filter[1] = FilterType::New();
	// Software Guide : EndCodeSnippet


	//  Software Guide : BeginLatex
	//
	//  The order of the derivative is selected with the \code{SetOrder()}
	//  method.  The direction along which the derivative will be computed is
	//  selected with the \code{SetDirection()} method.
	//
	//  \index{itk::DerivativeImageFilter!SetOrder()}
	//  \index{itk::DerivativeImageFilter!SetDirection()}
	//
	//  Software Guide : EndLatex 

	// Software Guide : BeginCodeSnippet
	/*
	filter->SetOrder(     atoi( argv[4] ) );
	filter->SetDirection( atoi( argv[5] ) );
	*/

	for(int i=0; i<2; i++)
	{
		filter[i]->SetOrder( 1 );
		filter[i]->SetDirection( i );
	}

	// Software Guide : EndCodeSnippet


	//  Software Guide : BeginLatex
	//
	//  The input to the filter can be taken from any other filter, for example
	//  a reader. The output can be passed down the pipeline to other filters,
	//  for example, a writer. An update call on any downstream filter will
	//  trigger the execution of the derivative filter.
	//
	//  \index{itk::DerivativeImageFilter!SetInput()}
	//  \index{itk::DerivativeImageFilter!GetOutput()}
	//
	//  Software Guide : EndLatex 


	// Software Guide : BeginCodeSnippet
	filter[0]->SetInput( reader->GetOutput() );
	filter[1]->SetInput( filter[0]->GetOutput() );
	writer->SetInput( filter[1]->GetOutput() );
	writer->Update();
	// Software Guide : EndCodeSnippet


	//  Software Guide : BeginLatex
	// 
	// \begin{figure}
	// \center
	// \includegraphics[width=0.44\textwidth]{BrainProtonDensitySlice.eps}
	// \includegraphics[width=0.44\textwidth]{DerivativeImageFilterOutput.eps}
	// \itkcaption[Effect of the Derivative filter.]{Effect of the Derivative filter
	// on a slice from a MRI proton density brain image.}
	// \label{fig:DerivativeImageFilterOutput}
	// \end{figure}
	//
	//  Figure \ref{fig:DerivativeImageFilterOutput} illustrates the effect of
	//  the DerivativeImageFilter on a slice of MRI brain image. The derivative
	//  is taken along the $x$ direction.  The sensitivity to noise in the image
	//  is evident from this result.
	//
	//  Software Guide : EndLatex 


	typedef itk::Image< unsigned char, Dimension >  WriteImageType;

	typedef itk::RescaleIntensityImageFilter< 
		OutputImageType,
		WriteImageType >    NormalizeFilterType;

	typedef itk::ImageFileWriter< WriteImageType >       NormalizedWriterType;

	NormalizeFilterType::Pointer normalizer = NormalizeFilterType::New();
	NormalizedWriterType::Pointer normalizedWriter = NormalizedWriterType::New();

	normalizer->SetInput( filter[1]->GetOutput() );
	normalizedWriter->SetInput( normalizer->GetOutput() );

	normalizer->SetOutputMinimum(   0 );
	normalizer->SetOutputMaximum( 255 );

	normalizedWriter->SetFileName( argv[3] );
	normalizedWriter->Update();

	return EXIT_SUCCESS;

#endif


#ifdef MODE_SELFTEST

	int xdim, ydim, zdim;

	xdim = ydim = zdim = 32;
	for(int nTest=0; nTest<4; nTest++)
	{
		char input[100], itkout[100], cudaout[100];

		sprintf(input, "input_%dx%dx%d.nrrd", xdim, ydim, zdim);
		sprintf(itkout, "out_itk_%dx%dx%d.nrrd", xdim, ydim, zdim);
		sprintf(cudaout, "out_cuda_%dx%dx%d.nrrd", xdim, ydim, zdim);

		//
		// create a random 3D volume
		//
		ItkImagePointer vol_in, vol_out[2];

		// set size & region
		ItkImageType::IndexType start;
		start[0] = 0; start[1] = 0; start[2] = 0;
		ItkImageType::SizeType size;
		size[0] = xdim; size[1] = ydim; size[2] = zdim;
		ItkImageType::RegionType region;
		region.SetSize( size );
		region.SetIndex( start );

		// create
		vol_in = ItkImageType::New();
		vol_in->SetRegions( region );
		vol_in->Allocate();

		// initialize volume with random numbers
		srand( time(0) );
		pixeltype v = 0;
		ImageRegionIterator<ItkImageType> it( vol_in, vol_in->GetRequestedRegion() );
		for(it = it.Begin(); !it.IsAtEnd(); ++it)
		{         
			pixeltype val = (pixeltype)(rand()%256);
			it.Set( val ); 
			/*
			it.Set(v);
			v += 1.0;
			*/
		}

		int direction = 1;

		typedef itk::DerivativeImageFilter< ItkImageType, ItkImageType >  FilterType;
		FilterType::Pointer filter[2];

		std::cout << "------------------------------------------------------" << std::endl;
		std::cout << " Input Size : " << xdim << "x" << ydim << "x" << zdim << std::endl;
		std::cout << "------------------------------------------------------" << std::endl;

		// CUDA
		std::cout << "CUDA Derivative Filter" << std::endl;
		putenv("ITK_CUDA=1");
		//putenv("CUDA_LAUNCH_BLOCKING=1");
		filter[1] = FilterType::New();
		filter[1]->SetOrder(1);
		filter[1]->SetDirection(direction);
		filter[1]->SetInput( vol_in );
		filter[1]->Update();
		vol_out[1] = filter[1]->GetOutput();
		/*
		// check the value
		ImageRegionIterator<ItkImageType> nit( vol_out[1], vol_out[1]->GetRequestedRegion() );
		for(nit = nit.Begin(); !nit.IsAtEnd(); ++nit)
		{
		std::cout << "Returned Value : " << nit.Get() << std::endl;
		}

		std::cout << "------------------------------------------------------" << std::endl;
		*/
		std::cout << "------------------------------------------------------" << std::endl;
		// ITK
		std::cout << "ITK Derivative Filter" << std::endl;
		putenv("ITK_CUDA=0"); 
		filter[0] = FilterType::New();
		filter[0]->SetOrder(1);
		filter[0]->SetDirection(direction);
		filter[0]->SetInput( vol_in );
		filter[0]->Update();
		vol_out[0] = filter[0]->GetOutput();

		std::cout << "------------------------------------------------------" << std::endl;

		// compute the error
		double err = 0;
		ImageRegionConstIteratorWithIndex<ItkImageType> it1( vol_out[0], vol_out[0]->GetRequestedRegion() );
		ImageRegionConstIteratorWithIndex<ItkImageType> it2( vol_out[1], vol_out[1]->GetRequestedRegion() );
		for(it1 = it1.Begin(), it2 = it2.Begin(); !it1.IsAtEnd(), !it2.IsAtEnd(); ++it1, ++it2)
		{
			ItkImageType::IndexType idx;
			idx = it1.GetIndex();

			double val[2], terr;
			val[0] = (double)it1.Get(); // itk
			val[1] = (double)it2.Get(); // CUDA
			terr = fabs(val[0] - val[1]);
			err += terr;

			if(terr > 0.01)
			{
				std::cout << "Error : " << terr << std::endl;
			}

		}
		std::cout << "Total error (L1) : " << err / (double)(xdim*ydim*zdim*255) << std::endl;
		std::cout << "======================================================" << std::endl << std::endl;

		// enlarge volume size
		xdim *= 2;
		ydim *= 2;
		zdim *= 2;
	}

#endif


}

