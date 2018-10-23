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
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkTimeProbe.h"

typedef float pixeltype; // unsigned char
typedef itk::Image<pixeltype, 3> ItkImageType;
typedef ItkImageType::Pointer ItkImagePointer;

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
 
  putenv("ITK_CUDA=1");
  int xdim, ydim;
 
  /*
  if( argc < 5 )
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "input output variance  maxKernelWidth " << std::endl;
    return EXIT_FAILURE;
  }
  */

  argv[1] = "C:/Work/proj/InsightToolkit-3.2.0/Examples/Data/BrainProtonDensitySlice256x256.png";
  argv[2] = "output_cuda_gaussian.png";
  argv[3] = "6";
  argv[4] = "40";

  typedef    float    InputPixelType;
  typedef    float    OutputPixelType;

  typedef itk::Image< InputPixelType,  2 >   InputImageType;
  typedef itk::Image< OutputPixelType, 2 >   OutputImageType;
  typedef itk::ImageFileReader< InputImageType >  ReaderType;
  typedef itk::DiscreteGaussianImageFilter<
                 InputImageType, OutputImageType >  FilterType;

  FilterType::Pointer filter = FilterType::New();
  
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( argv[1] );

  filter->SetInput( reader->GetOutput() );
  
  double gaussianVariance = atof( argv[3] );
  unsigned int maxKernelWidth = atoi( argv[4] );
  maxKernelWidth = ceil(gaussianVariance*5.0)*2 + 1;

  filter->SetVariance( gaussianVariance );
  filter->SetMaximumKernelWidth( maxKernelWidth );
  filter->Update();
  
  typedef unsigned char WritePixelType;
  typedef itk::Image< WritePixelType, 2 > WriteImageType;
  typedef itk::RescaleIntensityImageFilter< 
               OutputImageType, WriteImageType > RescaleFilterType;
  RescaleFilterType::Pointer rescaler = RescaleFilterType::New();

  rescaler->SetOutputMinimum(   0 );
  rescaler->SetOutputMaximum( 255 );

  typedef itk::ImageFileWriter< WriteImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[2] );
 
  rescaler->SetInput( filter->GetOutput() );
  writer->SetInput( rescaler->GetOutput() );
  writer->Update();
  
  return EXIT_SUCCESS;

#endif


#ifdef MODE_SELFTEST

   putenv("ITK_CUDA_TIME=0");
  int xdim, ydim, zdim, gaussianVariance;

  for(gaussianVariance=1; gaussianVariance<=8; gaussianVariance*=2)
  {
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
    
      // Create 2 filters, one for ITK and the other for CUDA
      typedef itk::DiscreteGaussianImageFilter< ItkImageType, ItkImageType >  FilterType;
   
      FilterType::Pointer filter[2];
     
      unsigned int maxKernelWidth = ceil(gaussianVariance*5.0)*2 + 1;
 
      std::cout << "------------------------------------------------------" << std::endl;
      std::cout << " Variance : " << gaussianVariance << std::endl;
      std::cout << " Input Size : " << xdim << "x" << ydim << "x" << zdim << std::endl;
      std::cout << "------------------------------------------------------" << std::endl;
      
      // ITK
      {
        std::cout << "ITK Gaussian Filter" << std::endl;
        putenv("ITK_CUDA=0"); 
        filter[0] = FilterType::New();
        filter[0]->SetVariance( gaussianVariance );
        filter[0]->SetMaximumKernelWidth( maxKernelWidth );
        filter[0]->SetInput( vol_in );
       
        itk::TimeProbe timer;
        timer.Start();        
        filter[0]->Update();        
        timer.Stop();  
        std::cout << std::setprecision(3) << "Time: " << timer.GetMeanTime() << std::endl;
   
        vol_out[0] = filter[0]->GetOutput();
        std::cout << "------------------------------------------------------" << std::endl;
      }


      // CUDA
      std::cout << "CUDA Gaussian Filter" << std::endl;
      putenv("ITK_CUDA=1");
     
      if(nTest == 0) // GPU warming-up for accurate time measurement
      {
        putenv("ITK_CUDA_TIME=0");
        filter[1] = FilterType::New();
        filter[1]->SetVariance( gaussianVariance );
        filter[1]->SetMaximumKernelWidth( maxKernelWidth );
        filter[1]->SetInput( vol_in );
        filter[1]->Update();
      }

      {
        //putenv("ITK_CUDA_TIME=1");
        filter[1] = FilterType::New();
        filter[1]->SetVariance( gaussianVariance );
        filter[1]->SetMaximumKernelWidth( maxKernelWidth );
        filter[1]->SetInput( vol_in );
   
        itk::TimeProbe timer;
        timer.Start();        
        filter[1]->Update();        
        timer.Stop();  
        std::cout << std::setprecision(3) << "Time: " << timer.GetMeanTime() << std::endl;
   
        vol_out[1] = filter[1]->GetOutput();
      }

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

      // compute the error
      double err = 0;
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
          std::cout << "Error : " << terr << std::endl;
        }
        */
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

}

