/*=========================================================================
  itkTestHessian.cxx
  Source code from 
  http://public.kitware.com/pipermail/insight-users/2007-May/022200.html
=========================================================================*/

#include <iomanip>
#include "itkImage.h"
#include "itkNumericTraits.h"
#include "itkCommand.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkTimeProbe.h"

#include <time.h>
#include <stdlib.h>
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"


// Declare general types
const unsigned int Dimension = 3;
typedef float PixelType;
typedef itk::Image< PixelType, Dimension > ImageType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageFileWriter< ImageType > WriterType;
typedef itk::HessianRecursiveGaussianImageFilter< ImageType > HessianFilterType;
typedef HessianFilterType::OutputImageType OutputImageType;

// Declare a command to display the progress in a nice block format
class ProgressCommand : public itk::Command
{
public:
  typedef  ProgressCommand   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );

protected:
  ProgressCommand() { m_LastProgress = -1; };
  float m_LastProgress;

public:

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
      const itk::ProcessObject* process = dynamic_cast< const itk::ProcessObject* >( object );

      if( ! itk::ProgressEvent().CheckEvent( &event ) )
        return;

      int fprogress = (process->GetProgress() * 100.0);
      int progress = (int)(process->GetProgress() * 100.0);
      if ((int)m_LastProgress == progress)
          return;
      if ((int)m_LastProgress != (progress - 1))
      {
          std::cout << std::setfill('0') << std::setw(3) << (progress - 1) << " ";
          if (fprogress > 0.0 && (progress - 1) % 10 == 0)
            std::cout << std::endl;
      }
      if (fprogress > 0.0 && fprogress <= 100.0)
          std::cout << std::setfill('0') << std::setw(3) << progress << " ";
      if (fprogress > 0.0 && progress % 10 == 0)
          std::cout << std::endl;
      m_LastProgress = fprogress;
  }  
};

int main(int argc, char* argv[])
{
    int xdim, ydim, zdim, nTest;
    
    xdim = ydim = zdim = 32;
    nTest = 4;

    // Display header
    std::cout << "Hessian Test -------------------------" << std::endl;

    // Setup the algorithm parameters
    unsigned int argn = 1;
    //char* InputFilename        = argv[argn++];
    bool NormalizeAcrossScale  = true; // (bool)atoi( argv[argn++] );
    double Sigma               = 2.0;  //atof( argv[argn++] );

    // Display parameters
    //std::cout << "InputFilename:" << InputFilename << std::endl;
    std::cout << "NormalizeAcrossScale:" << NormalizeAcrossScale << std::endl;
    std::cout << "Sigma:" << Sigma << std::endl;
    std::cout << "------------------------------------" << std::endl;
   
    try
    {      
      for(int i=0; i<nTest; i++)
      {
        std::cout << "Image size : " << xdim << "x" << ydim << "x" << zdim << std::endl;

        //
        // create a random 3D volume
        //
        ImageType::Pointer vol_in;

        // set size & region
        ImageType::IndexType start;
        start[0] = 0; start[1] = 0; start[2] = 0;
        ImageType::SizeType size;
        size[0] = xdim; size[1] = ydim; size[2] = zdim;
        ImageType::RegionType region;
        region.SetSize( size );
        region.SetIndex( start );

        // create
        vol_in = ImageType::New();
        vol_in->SetRegions( region );
        vol_in->Allocate();

        // initialize volume with random numbers
        srand( time(0) );
        PixelType v = 0;
        itk::ImageRegionIterator<ImageType> it( vol_in, vol_in->GetRequestedRegion() );
        for(it = it.Begin(); !it.IsAtEnd(); ++it)
        {         
          PixelType val = (PixelType)(rand()%256);
          it.Set( val ); 
        }


        //
        // 1. CPU Hessian
        //
        {
          // Setup Hessian
          std::cout << "Computing Hessian using ITK..." << std::endl;
          putenv("ITK_CUDA=0");
          HessianFilterType::Pointer filterHessian = HessianFilterType::New();
          filterHessian->SetInput( vol_in );
          filterHessian->SetSigma( Sigma );
          filterHessian->SetNormalizeAcrossScale( NormalizeAcrossScale );
        
          // Compute and time hessian
          itk::TimeProbe time;
          time.Start();
          filterHessian->Update( );
          time.Stop();
          std::cout << std::setprecision(3) << "Time: " << time.GetMeanTime() << std::endl;
        }

        //
        // 2. GPU Hessian
        //
        putenv("ITK_CUDA=1");
        std::cout << "-----------------------" << std::endl;
        std::cout << "Computing Hessian using CUDA ITK..." << std::endl;

        if(i==0) // GPU warming-up
        {
          // Setup dummy Hessian filter and run
          HessianFilterType::Pointer filterHessianCUDA = HessianFilterType::New();
          filterHessianCUDA->SetInput( vol_in );
          filterHessianCUDA->SetSigma( Sigma );
          filterHessianCUDA->SetNormalizeAcrossScale( NormalizeAcrossScale );
          filterHessianCUDA->Update( );         
        }

        { // explicit scope for ITK smart pointer
         
          // Setup Hessian        
          HessianFilterType::Pointer filterHessianCUDA = HessianFilterType::New();
          filterHessianCUDA->SetInput( vol_in );
          filterHessianCUDA->SetSigma( Sigma );
          filterHessianCUDA->SetNormalizeAcrossScale( NormalizeAcrossScale );
         
          // Compute and time hessian
          itk::TimeProbe timeCUDA;
          timeCUDA.Start();
          filterHessianCUDA->Update( );
          timeCUDA.Stop();
          std::cout << std::setprecision(3) << "Time: " << timeCUDA.GetMeanTime() << std::endl;
        }

        /*
        // print out results
        OutputImageType::Pointer vol_out[2];
        vol_out[0] = filterHessian->GetOutput();
        vol_out[1] = filterHessianCUDA->GetOutput();
        OutputImageType::PixelType pix[2];
        itk::ImageRegionIterator<OutputImageType> oitITK( vol_out[0], vol_out[0]->GetRequestedRegion() );
        itk::ImageRegionIterator<OutputImageType> oitCUDA( vol_out[1], vol_out[1]->GetRequestedRegion() );
       
        for(oitITK = oitITK.Begin(); !oitITK.IsAtEnd(); ++oitITK, oitCUDA++)
        {         
          pix[0] = oitITK.Get();
          pix[1] = oitCUDA.Get();

          // print
          std::cout << "ITK : ";
          for(int i=0; i<6; i++)
          {
            std::cout << std::setprecision(6) << pix[0][i] << ", ";
          }
          std::cout << std::endl;

          std::cout << "CUDA : ";
          for(int i=0; i<6; i++)
          {
            std::cout << std::setprecision(6) << pix[1][i] << ", ";
          }

          std::cout << std::endl;
          std::cout << std::endl;
        }
        */

        std::cout << "------------------------------------" << std::endl;

        xdim *= 2;
        ydim *= 2;
        zdim *= 2;
      }
    }

    catch (itk::ExceptionObject & err)
    {
        std::cout << "ExceptionObject caught !" << std::endl;
        std::cout << err << std::endl;
        return EXIT_FAILURE;
    }

    //Return
    return EXIT_SUCCESS;
}
