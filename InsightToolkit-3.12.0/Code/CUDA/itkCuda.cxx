#include "itkCuda.h"

namespace itk
{

void initCUDA()
{
  static bool initialized = false;

  if(!initialized)
  {
    std::cout << "Initializing CUDA device... " << std::endl;
    CUdevice dev;
    CUcontext con;
    int count=0;
    cuInit(0) ;
    cuDeviceGetCount(&count) ;
    if( count <= 0 )
    {
      printf("No compute devices found\n");
    }
    else
    {
      //get the compute device
      cuDeviceGet(&dev, 0);
      //create the context
      cuCtxCreate(&con, 0, dev);
      initialized = true;
      printf("Done!\n");
    }
/*
    // warm up memory
    float *d_t, h_t;
    h_t = 1;
    cudaMalloc((void**) &d_t, sizeof(float));
    cudaMemcpy(d_t, &h_t, sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(d_t);
*/
  }
}

} // namespace