#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include <iomanip>
#include <time.h>
#include <stdlib.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkMeanImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkTimeProbe.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"

//
// MODE_SELFTEST : run filter on various size datasets and calculate timing
// MODE_INPUT    : run filters on the input image/volume
//
#define MODE_SELFTEST //MODE_INPUT // 
#define ANISO_FILTER //MEDIAN_FILTER // ANISO_FILTER //MEAN_FILTER //

   

// Filter type
#ifdef MEAN_FILTER
  char filtername[] = "Mean filter";
  typedef float pixeltype;
  typedef itk::Image<pixeltype, 3> ItkImageType;
  typedef itk::MeanImageFilter< ItkImageType, ItkImageType >  FilterType;
#elif defined ( MEDIAN_FILTER )
  char filtername[] = "Median filter";
  typedef unsigned char pixeltype;
  typedef itk::Image<pixeltype, 3> ItkImageType;
  typedef itk::MedianImageFilter< ItkImageType, ItkImageType >  FilterType;
#elif defined ( ANISO_FILTER )
  char filtername[] = "Anisotropic diffusion filter";
  typedef float pixeltype;
  typedef itk::Image<pixeltype, 3> ItkImageType;
  typedef itk::GradientAnisotropicDiffusionImageFilter< ItkImageType, ItkImageType > FilterType;
#else
  char filtername[] = "Mean filter";
  typedef float pixeltype;
  typedef itk::Image<pixeltype, 3> ItkImageType;
  typedef itk::MeanImageFilter< ItkImageType, ItkImageType >  FilterType;
#endif




typedef ItkImageType::Pointer ItkImagePointer;

using namespace itk;

int main( int argc, char * argv[] )
{

#ifdef MODE_INPUT

  int xdim, ydim;
 
  /*
  if( argc < 6 )
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "input outputITK outputCUDA x-radius y-radius " << std::endl;
    return EXIT_FAILURE;
  }
  */

  argv[1] = "C:/Work/proj/InsightToolkit-3.2.0/Examples/Data/BrainProtonDensitySlice256x256.png";
  argv[2] = "output_itk_new.png";
  argv[3] = "output_CUDA_new.png";
  argv[4] = "3"; // small error when the filter size is 5
  argv[5] = "3";


  std::cout << "Input image : " << argv[1] << ", Kernel size : " << argv[4] << "x" << argv[5] << std::endl;

  typedef   unsigned char  InputPixelType;
  typedef   unsigned char  OutputPixelType;

  typedef itk::Image< InputPixelType,  2 >   InputImageType;
  typedef itk::Image< OutputPixelType, 2 >   OutputImageType;
 
  typedef itk::ImageFileReader< InputImageType  >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  typedef itk::MeanImageFilter< InputImageType, OutputImageType >  FilterType;
  
  InputImageType::SizeType indexRadius;  
  indexRadius[0] = atoi(argv[4]); // radius along x
  indexRadius[1] = atoi(argv[5]); // radius along y

  reader->SetFileName( argv[1] );

  std::cout << "------------------------------------------------------" << std::endl;
  
  

  // Create 2 filters, one for ITK and the other for CUDA
  FilterType::Pointer filter[2];

  // ITK
  std::cout << "ITK Mean Filter" << std::endl;
  putenv("ITK_CUDA=0"); 
  filter[0] = FilterType::New();
  filter[0]->SetRadius( indexRadius );
  filter[0]->SetInput( reader->GetOutput() );
  filter[0]->Update();

  std::cout << "------------------------------------------------------" << std::endl;

  // CUDA
  std::cout << "CUDA Mean Filter" << std::endl;
  putenv("ITK_CUDA=1");
  putenv("CUDA_LAUNCH_BLOCKING=1");
  filter[1] = FilterType::New();
  filter[1]->SetRadius( indexRadius );
  filter[1]->SetInput( reader->GetOutput() );
  filter[1]->Update();

  // save 
  writer->SetFileName( argv[2] );
  writer->SetInput( filter[0]->GetOutput() );
  writer->Update();

  writer->SetFileName( argv[3] );
  writer->SetInput( filter[1]->GetOutput() );
  writer->Update();
 
  // get the size of the image
  OutputImageType::SizeType imageSize = (filter[0]->GetOutput())->GetLargestPossibleRegion().GetSize();
  xdim = imageSize[0];
  ydim = imageSize[1];

  // compute the error
  double err = 0;
  ImageRegionConstIterator<OutputImageType> it1( filter[0]->GetOutput(), (filter[0]->GetOutput())->GetRequestedRegion() );
  ImageRegionConstIterator<OutputImageType> it2( filter[1]->GetOutput(), (filter[1]->GetOutput())->GetRequestedRegion() );
  for(it1 = it1.Begin(), it2 = it2.Begin(); !it1.IsAtEnd(), !it2.IsAtEnd(); ++it1, ++it2)
  {
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
  std::cout << "Total error (L1) : " << err / (double)(xdim*ydim) << std::endl;


  std::cout << "------------------------------------------------------" << std::endl;

  return EXIT_SUCCESS;

#endif


#ifdef MODE_SELFTEST
  
  int xdim, ydim, zdim;

  putenv("ITK_CUDA_TIME=0");

  for(int kernelsize=1, niter = 2; kernelsize<=4; kernelsize++, niter*=2)
  {
    xdim = ydim = zdim = 32;
    for(int nTest=0; nTest<4; nTest++)
  
  /*
  for(int kernelsize=3; kernelsize<=3; kernelsize++)
  {
    xdim = ydim = 256; zdim = 100;
    for(int nTest=0; nTest<1; nTest++)
    */
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
      unsigned int i=0;
      for(it = it.Begin(); !it.IsAtEnd(); ++it)
      {            
        pixeltype val = (rand()%256);//(pixeltype)i;//
        it.Set( val ); i++;
      }
/*
      for(int z=0; z<zdim; z++)
      {
        for(int y=0; y<ydim; y++)
        {
          for(int x=0; x<xdim; x++)
          {
            ItkImageType::IndexType idx;
            idx[0] = x;
            idx[1] = y;
            idx[2] = z;

            pixeltype val = vol_in->GetPixel(idx);

            std::cout << val << std::endl;
          }
        }
      }
*/

         
     

#ifdef ANISO_FILTER

      std::cout << "------------------------------------------------------" << std::endl;
      std::cout << "Iteration   : " << niter << std::endl;
      std::cout << "Image size  : " << xdim << "x" << ydim << "x" << zdim << std::endl;
      std::cout << "------------------------------------------------------" << std::endl;
      
      std::cout << "ITK " << filtername << std::endl;
      {
        putenv("ITK_CUDA=0"); 
        
        FilterType::Pointer filter;
        filter = FilterType::New();
        filter->SetNumberOfIterations( niter );
        filter->SetTimeStep( 0.0625 );
        filter->SetConductanceParameter( 5.0 );
        filter->SetInput( vol_in );

        itk::TimeProbe timer;
        timer.Start();        
        filter->Update();        
        timer.Stop();  
        std::cout << std::setprecision(3) << "Time: " << timer.GetMeanTime() << std::endl;
   
        vol_out[0] = filter->GetOutput();
      }

      std::cout << "------------------------------------------------------" << std::endl;


      if(nTest==0) // GPU warming up
      {         
        putenv("ITK_CUDA=1");
        putenv("ITK_CUDA_TIME=0");
        FilterType::Pointer filter;
        filter = FilterType::New();
        filter->SetNumberOfIterations( 1 );
        filter->SetTimeStep( 0.0625 );
        filter->SetConductanceParameter( 5.0 );
        filter->SetInput( vol_in );
        filter->Update();
      }

      {
        // CUDA
        std::cout << "CUDA " << filtername << std::endl;
        putenv("ITK_CUDA=1");
        //putenv("ITK_CUDA_TIME=1");
        FilterType::Pointer filter;
        filter = FilterType::New();
        filter->SetNumberOfIterations( niter );
        filter->SetTimeStep( 0.0625 );
        filter->SetConductanceParameter( 5.0 );
        filter->SetInput( vol_in );
                   
        itk::TimeProbe timer;
        timer.Start();
        filter->Update();
        timer.Stop();  
        std::cout << std::setprecision(3) << "Time: " << timer.GetMeanTime() << std::endl;
   
        vol_out[1] = filter->GetOutput();
      }
     

#else

       ItkImageType::SizeType indexRadius;  
      indexRadius[0] = kernelsize;
      indexRadius[1] = kernelsize;
      indexRadius[2] = kernelsize;

      std::cout << "------------------------------------------------------" << std::endl;
      std::cout << "Kernel size : " << 2*kernelsize + 1 << "^3" << std::endl;
      std::cout << "Image size  : " << xdim << "x" << ydim << "x" << zdim << std::endl;
      std::cout << "------------------------------------------------------" << std::endl;
  

      std::cout << "ITK " << filtername << std::endl;
      {
        putenv("ITK_CUDA=0"); 
        
        FilterType::Pointer filter;
        filter = FilterType::New();
        filter->SetRadius( indexRadius );
        filter->SetInput( vol_in );

        itk::TimeProbe timer;
        timer.Start();        
        filter->Update();        
        timer.Stop();  
        std::cout << std::setprecision(3) << "Time: " << timer.GetMeanTime() << std::endl;
   
        vol_out[0] = filter->GetOutput();
      }

      std::cout << "------------------------------------------------------" << std::endl;


      if(nTest==0) // GPU warming up
      {         
        putenv("ITK_CUDA=1");
        putenv("ITK_CUDA_TIME=0");
        FilterType::Pointer filter;
        filter = FilterType::New();
        filter->SetRadius( indexRadius );
        filter->SetInput( vol_in );
        filter->Update();
      }

      {
        // CUDA
        std::cout << "CUDA " << filtername << std::endl;
        putenv("ITK_CUDA=1");
        //putenv("ITK_CUDA_TIME=1");
        FilterType::Pointer filter;
        filter = FilterType::New();
        filter->SetRadius( indexRadius );
        filter->SetInput( vol_in );

        itk::TimeProbe timer;
        timer.Start();
        filter->Update();
        timer.Stop();  
        std::cout << std::setprecision(3) << "Time: " << timer.GetMeanTime() << std::endl;
   
        vol_out[1] = filter->GetOutput();
      }
     
#endif


      std::cout << "------------------------------------------------------" << std::endl;
  

      // compute the error
      double err = 0;
      unsigned int idx = 0;
      ImageRegionConstIterator<ItkImageType> it1( vol_out[0], vol_out[0]->GetRequestedRegion() );
      ImageRegionConstIterator<ItkImageType> it2( vol_out[1], vol_out[1]->GetRequestedRegion() );
      for(it1 = it1.Begin(), it2 = it2.Begin(); !it1.IsAtEnd(), !it2.IsAtEnd(); ++it1, ++it2)
      {
        double val[2], terr;
        val[0] = (double)it1.Get(); // itk
        val[1] = (double)it2.Get(); // CUDA
        terr = fabs(val[0] - val[1]);
        err += terr;
        /*
        if(terr > 0.01)
        {
			std::cout << "idx : " << idx << ", ITK : " << val[0] << ", CUDA : " << val[1] << ", Error : " << terr << std::endl;
        }
		*/
        
        idx++;   
      }
      std::cout << "Total error (L1) : " << err / (double)(xdim*ydim*zdim*255) << std::endl;
      std::cout << "======================================================" << std::endl << std::endl;


      /*
      // save volumes
      typedef itk::ImageFileWriter< ItkImageType >  WriterType;
      WriterType::Pointer writer = WriterType::New();

      writer->SetFileName( input );
      writer->SetInput( vol_in );
      writer->Update();

      writer->SetFileName( itkout );
      writer->SetInput( vol_out[0] );
      writer->Update();

      writer->SetFileName( cudaout );
      writer->SetInput( vol_out[1] );
      writer->Update();
      */

      // enlarge volume size
      xdim *= 2;
      ydim *= 2;
      zdim *= 2;
    }
  }
#endif

  std::cout << "Test done!" << std::endl;
}
