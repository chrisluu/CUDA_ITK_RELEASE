#include "itkCudaCommon.h"

//#define USE_SHARED_MEMORY

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 4

//__global__ void kernel_GradientAnisotropicDiffusionFilter_F2F(float* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim, int nUpdatePerThread, float timestep, float conductance, int numiter);
__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F(float* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance);
__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_U2F(uchar* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance);
__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2U(float* _in, uchar* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance);
__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_U2U(uchar* _in, uchar* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance);
__global__ void kernel_SumGradientMagnitude_FLOAT(float* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim);
__global__ void kernel_SumGradientMagnitude_UCHAR(uchar* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim);


extern "C"
void cudaGradientAnisotropicDiffusionFilterKernelWrapper(void *_in, void *_out,
                                                         FILTERTYPE type, int dimension, int numiter,
                                                         AnisotropicDiffusionParameter param,
                                                         uint image_width, uint image_height, uint image_depth)
{
/*

  uint nGlobalIter = ceil((float)numiter / MAX_LOCAL_ITER);

  float *d_temp[2]; // temporary buffer

  // dynamic memory allocation
  if(nGlobalIter > 1)
  {
    cudaMalloc((void**)&d_temp[0], sizeof(float)*image_width*image_height*image_depth);
    CUT_CHECK_ERROR("Memory creation failed");
    if(nGlobalIter > 2)
    {
      cudaMalloc((void**)&d_temp[1], sizeof(float)*image_width*image_height*image_depth);
      CUT_CHECK_ERROR("Memory creation failed");
    }
  }


  // define grid / block dimension
  int blockWidth, blockHeight, blockDepth;
  int nBlockX, nBlockY, nBlockZ;

  // Setting block size
  if(dimension == 2) 
  {
    blockWidth  = 16;
    blockHeight = 16;
    blockDepth  = 1;

    image_depth = 1;  
  }
  else
  {
    blockWidth  = 4;
    blockHeight = 4;
    blockDepth  = 4;     
  }

  // compute how many blocks are needed
  nBlockX = ceil((float)image_width / (float)blockWidth);
  nBlockY = ceil((float)image_height / (float)blockHeight);
  nBlockZ = ceil((float)image_depth / (float)blockDepth);
 
  dim3 dimGrid(nBlockX,nBlockY*nBlockZ); // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight, blockDepth);
  dim3 imagedim(image_width,image_height,image_depth);
  dim3 griddim(nBlockX,nBlockY,nBlockZ);
  
  // create additional memory for temporary buffer 
  for(int i=numiter, j=0; i>0; i-=MAX_LOCAL_ITER, j++)
  {
    uint nLocalIter = min(i, MAX_LOCAL_ITER);
    dim3 sharedmemdim((nLocalIter*2)+blockWidth,(nLocalIter*2)+blockHeight,(nLocalIter*2)+blockDepth);
    if(dimension == 2) sharedmemdim.z = 1;
    uint sharedmemsize = 2*sizeof(float)*sharedmemdim.x*sharedmemdim.y*sharedmemdim.z;
    uint updateperthread = ceil((float)(sharedmemdim.x*sharedmemdim.y*sharedmemdim.z)/(float)(blockWidth*blockHeight*blockDepth));
  
    if(nGlobalIter < 2)
    {
      kernel_GradientAnisotropicDiffusionFilter_F2F<<<dimGrid,dimBlock,sharedmemsize>>>((float*)_in, (float*)_out, dimension, imagedim, griddim, updateperthread, timestep, conductance, nLocalIter);
    }
    else 
    {
      if(j == 0) // 1st iter
      {
        kernel_GradientAnisotropicDiffusionFilter_F2F<<<dimGrid,dimBlock,sharedmemsize>>>((float*)_in, (float*)d_temp[j%2], dimension, imagedim, griddim, updateperthread, timestep, conductance, nLocalIter);
      }
      else if(j < (nGlobalIter-1))
      {
        kernel_GradientAnisotropicDiffusionFilter_F2F<<<dimGrid,dimBlock,sharedmemsize>>>((float*)d_temp[(j+1)%2], (float*)d_temp[j%2], dimension, imagedim, griddim, updateperthread, timestep, conductance, nLocalIter);
      }
      else  // last iter
      {
        kernel_GradientAnisotropicDiffusionFilter_F2F<<<dimGrid,dimBlock,sharedmemsize>>>((float*)d_temp[(j+1)%2], (float*)_out, dimension, imagedim, griddim, updateperthread, timestep, conductance, nLocalIter);
      }
    }
  }


  // memory release
  if(nGlobalIter > 1)
  {
    cudaFree(d_temp[0]);
    CUT_CHECK_ERROR("Memory release failed");
    if(nGlobalIter > 2)
    {
      cudaFree(d_temp[1]);
      CUT_CHECK_ERROR("Memory release failed");
    }
  }

  */

  float m_K;     
  float *d_temp[2], *d_sumGradMag, *h_sumGradMag; // temporary buffer

  // dynamic memory allocation
  if(numiter > 1)
  {
    cudaMalloc((void**)&d_temp[0], sizeof(float)*image_width*image_height*image_depth);
    CUT_CHECK_ERROR("Memory creation failed");
    if(numiter > 2)
    {
      cudaMalloc((void**)&d_temp[1], sizeof(float)*image_width*image_height*image_depth);
      CUT_CHECK_ERROR("Memory creation failed");
    }
  }

  // define grid / block dimension
  int blockWidth, blockHeight, blockDepth;
  int nBlockX, nBlockY, nBlockZ;

  // Setting block size
  if(dimension == 2) 
  {
    blockWidth  = 16;
    blockHeight = 16;
    blockDepth  = 1;

    image_depth = 1;  
  }
  else
  {
    blockWidth  = BLOCK_DIM_X;
    blockHeight = BLOCK_DIM_Y;
    blockDepth  = BLOCK_DIM_Z;     
  }

  // compute how many blocks are needed
  nBlockX = (int) ceil((float)image_width / (float)blockWidth);
  nBlockY = (int) ceil((float)image_height / (float)blockHeight);
  nBlockZ = (int) ceil((float)image_depth / (float)blockDepth);
 
  cudaMalloc((void**)&d_sumGradMag, sizeof(float)*nBlockX*nBlockY*nBlockZ);
  h_sumGradMag = (float*)malloc(sizeof(float)*nBlockX*nBlockY*nBlockZ);

  dim3 dimGrid(nBlockX,nBlockY*nBlockZ); // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight, blockDepth);
  dim3 imagedim(image_width,image_height,image_depth);
  dim3 griddim(nBlockX,nBlockY,nBlockZ);
  dim3 sharedmemdim(blockWidth+2,blockHeight+2,blockDepth+2);
  if(dimension == 2) sharedmemdim.z = 1;
   

  // temporary
  param.m_GradientMagnitudeIsFixed = true;

   // create additional memory for temporary buffer 
  if(param.m_GradientMagnitudeIsFixed)
  {
    m_K = sqrt(2.0)*param.m_FixedAverageGradientMagnitude*param.m_ConductanceParameter;
  }
 
  if(type == FLOAT_TO_FLOAT)
  {
    float *inbuf, *outbuf;
    for(int i=0; i<numiter; i++)
    {
      if(numiter < 2)
      {
        inbuf  = (float*)_in;
        outbuf = (float*)_out;
        //kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F<<<dimGrid,dimBlock>>>((float*)_in, (float*)_out, dimension, imagedim, griddim, timestep, m_K);
      }
      else 
      {
        if(i == 0) // 1st iter
        {
          inbuf  = (float*)_in;
          outbuf = (float*)d_temp[0];
          //kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F<<<dimGrid,dimBlock>>>((float*)_in, (float*)d_temp[i%2], dimension, imagedim, griddim, timestep, m_K);
        }
        else if(i < (numiter-1))
        {
          inbuf  = (float*)d_temp[(i+1)%2];
          outbuf = (float*)d_temp[i%2];
          //kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F<<<dimGrid,dimBlock>>>((float*)d_temp[(i+1)%2], (float*)d_temp[i%2], dimension, imagedim, griddim, timestep, m_K);
        }
        else  // last iter
        {
          inbuf  = (float*)d_temp[(i+1)%2];
          outbuf = (float*)_out;
          //kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F<<<dimGrid,dimBlock>>>((float*)d_temp[(i+1)%2], (float*)_out, dimension, imagedim, griddim, timestep, m_K);
        }
      }

      
      if(!param.m_GradientMagnitudeIsFixed)
      {
        // compute Average Gradient Magnitude on every iteration
        kernel_SumGradientMagnitude_FLOAT<<<dimGrid,dimBlock>>>(inbuf, d_sumGradMag, dimension, imagedim, griddim);
        CUT_CHECK_ERROR("Kernel launch failed");
        cudaMemcpy(h_sumGradMag, d_sumGradMag, sizeof(float)*nBlockX*nBlockY*nBlockZ, cudaMemcpyDeviceToHost);
        double _sum = 0;
        for(uint i=0; i<nBlockX*nBlockY*nBlockZ; i++) _sum += h_sumGradMag[i];
        _sum /= (double)(imagedim.x*imagedim.y*imagedim.z);
        m_K = sqrt(2.0)*_sum*param.m_ConductanceParameter;
      }
      
      kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F<<<dimGrid,dimBlock>>>(inbuf, outbuf, dimension, imagedim, griddim, param.m_TimeStep, m_K);
       
      CUT_CHECK_ERROR("Kernel launch failed");
    }
  }
  else if(type == UCHAR_TO_UCHAR)
  {
    float *inbuf, *outbuf;
    
    for(int i=0; i<numiter; i++)
    {
      if(numiter > 1)
      {
        if(i == 0) // 1st iter
        {         
          outbuf = (float*)d_temp[0];
        }
        else if(i < (numiter-1))
        {
          inbuf  = (float*)d_temp[(i+1)%2];
          outbuf = (float*)d_temp[i%2];
        }
        else  // last iter
        {
          inbuf  = (float*)d_temp[(i+1)%2];        
        }
      }


      // compute Average Gradient Magnitude on every iteration
      if(!param.m_GradientMagnitudeIsFixed)
      {
        if(i==0)
        {
          kernel_SumGradientMagnitude_UCHAR<<<dimGrid,dimBlock>>>((uchar*)_in, d_sumGradMag, dimension, imagedim, griddim);
        }
        else
        {
          kernel_SumGradientMagnitude_FLOAT<<<dimGrid,dimBlock>>>(inbuf, d_sumGradMag, dimension, imagedim, griddim);
        }

        CUT_CHECK_ERROR("Kernel launch failed");
        cudaMemcpy(h_sumGradMag, d_sumGradMag, sizeof(float)*nBlockX*nBlockY*nBlockZ, cudaMemcpyDeviceToHost);
        double _sum = 0;
        for(uint i=0; i<nBlockX*nBlockY*nBlockZ; i++)
        {
          double val = h_sumGradMag[i];
          if(val > 50000) 
          {
            int bx, by, bz;
            bz = i / (nBlockX*nBlockY);
            by = (i - bz*(nBlockX*nBlockY)) / nBlockX;
            bx = i - bz*(nBlockX*nBlockY) - by*nBlockX;
            printf("Block idx : (%d,%d,%d), val : %f\n", bx, by, bz, val);
          }
          _sum += val / (double)(imagedim.x*imagedim.y*imagedim.z);
        }
        //_sum /= (double)(imagedim.x*imagedim.y*imagedim.z);
        m_K = sqrt(2.0)*_sum*param.m_ConductanceParameter;
      }
      
      if(numiter < 2)
      {
         kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_U2U<<<dimGrid,dimBlock>>>((uchar*)_in, (uchar*)_out, dimension, imagedim, griddim, param.m_TimeStep, m_K);
      }
      else
      {
        if(i == 0)
        {
          kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_U2F<<<dimGrid,dimBlock>>>((uchar*)_in, outbuf, dimension, imagedim, griddim, param.m_TimeStep, m_K);
        }
        else if(i == numiter-1)
        {
          kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2U<<<dimGrid,dimBlock>>>(inbuf, (uchar*)_out, dimension, imagedim, griddim, param.m_TimeStep, m_K);        
        }
        else
        {
          kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F<<<dimGrid,dimBlock>>>(inbuf, outbuf, dimension, imagedim, griddim, param.m_TimeStep, m_K);
        }
      }      
         
      CUT_CHECK_ERROR("Kernel launch failed");
    }
  }
  else
  {
    printf("ERROR : Not supported volume type\n");
    assert(false);
    exit(0);
  }


   // memory release
  if(numiter > 1)
  {
    cudaFree(d_temp[0]);
    CUT_CHECK_ERROR("Memory release failed");
    if(numiter > 2)
    {
      cudaFree(d_temp[1]);
      CUT_CHECK_ERROR("Memory release failed");
    }
  }
  cudaFree(d_sumGradMag);
  free(h_sumGradMag);
}



//
// Kernel
//
__global__ 
void kernel_SumGradientMagnitude_FLOAT(float* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim)
{
  dim3 bidx;
  uint idx, idy, idz, tid;

  bidx.x = blockIdx.x;
  bidx.z = blockIdx.y / griddim.y;
  bidx.y = blockIdx.y - bidx.z*griddim.y;

  idx = bidx.x*blockDim.x + threadIdx.x;
  idy = bidx.y*blockDim.y + threadIdx.y;
  idz = bidx.z*blockDim.z + threadIdx.z;
  
  __shared__ float shmem[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size
  __shared__ float grmag[BLOCK_DIM_X*BLOCK_DIM_Y*BLOCK_DIM_Z];  // magnitude of gradient 
 
  dim3 cidx;
  cidx.x = threadIdx.x + 1;
  cidx.y = threadIdx.y + 1;
  cidx.z = threadIdx.z + 1;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    //
    // load shared memory
    //
    shmem[cidx.x][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + idx];

    // loading neighbors : Neumann boundary condition
    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem[0][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem[blockDim.x + 1][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,0) - 1;
      shmem[cidx.x][0][cidx.z] = _in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem[cidx.x][blockDim.y + 1][cidx.z] = _in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,0) - 1;
      shmem[cidx.x][cidx.y][0] = _in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem[cidx.x][cidx.y][blockDim.z + 1] = _in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }       
  }

  __syncthreads();

  //
  // compute magnitude of gradient
  //
  tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    float gradX, gradY, gradZ;
    gradX = (shmem[cidx.x+1][cidx.y][cidx.z] - shmem[cidx.x-1][cidx.y][cidx.z])/2.0f;
    gradY = (shmem[cidx.x][cidx.y+1][cidx.z] - shmem[cidx.x][cidx.y-1][cidx.z])/2.0f;
    gradZ = (shmem[cidx.x][cidx.y][cidx.z+1] - shmem[cidx.x][cidx.y][cidx.z-1])/2.0f;

    grmag[tid] = sqrt(gradX*gradX + gradY*gradY + gradZ*gradZ);
  }
  else
  {
    grmag[tid] = 0.0f;
  }

  __syncthreads();


  uint halfThread = (blockDim.x * blockDim.y * blockDim.z) >> 1;

  // reduction
  for(uint i=halfThread; i>0; i = i>>1)
  {
    if(tid < i)
    {
      grmag[tid] = grmag[tid] + grmag[tid + i];
    }
  }

  __syncthreads();

  _out[(bidx.z*griddim.y + bidx.y)*griddim.x + bidx.x] = grmag[0];
}


__global__ 
void kernel_SumGradientMagnitude_UCHAR(uchar* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim)
{
  dim3 bidx;
  uint idx, idy, idz, tid;

  bidx.x = blockIdx.x;
  bidx.z = blockIdx.y / griddim.y;
  bidx.y = blockIdx.y - bidx.z*griddim.y;

  idx = bidx.x*blockDim.x + threadIdx.x;
  idy = bidx.y*blockDim.y + threadIdx.y;
  idz = bidx.z*blockDim.z + threadIdx.z;
  
  __shared__ float shmem[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size
  __shared__ float grmag[BLOCK_DIM_X*BLOCK_DIM_Y*BLOCK_DIM_Z];  // magnitude of gradient 
 
  dim3 cidx;
  cidx.x = threadIdx.x + 1;
  cidx.y = threadIdx.y + 1;
  cidx.z = threadIdx.z + 1;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    //
    // load shared memory
    //
    shmem[cidx.x][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + idx];

    // loading neighbors : Neumann boundary condition
    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem[0][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem[blockDim.x + 1][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,0) - 1;
      shmem[cidx.x][0][cidx.z] = (float)_in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem[cidx.x][blockDim.y + 1][cidx.z] = (float)_in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,0) - 1;
      shmem[cidx.x][cidx.y][0] = (float)_in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem[cidx.x][cidx.y][blockDim.z + 1] = (float)_in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }       
  }

  __syncthreads();

  //
  // compute magnitude of gradient
  //
  tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    float gradX, gradY, gradZ;
    gradX = (shmem[cidx.x+1][cidx.y][cidx.z] - shmem[cidx.x-1][cidx.y][cidx.z])/2.0f;
    gradY = (shmem[cidx.x][cidx.y+1][cidx.z] - shmem[cidx.x][cidx.y-1][cidx.z])/2.0f;
    gradZ = (shmem[cidx.x][cidx.y][cidx.z+1] - shmem[cidx.x][cidx.y][cidx.z-1])/2.0f;

    grmag[tid] = sqrt(gradX*gradX + gradY*gradY + gradZ*gradZ);
  }
  else
  {
    grmag[tid] = 0.0f;
  }

  __syncthreads();


  uint halfThread = (blockDim.x * blockDim.y * blockDim.z) >> 1;

  // reduction
  for(uint i=halfThread; i>0; i = i>>1)
  {
    if(tid < i)
    {
      grmag[tid] = grmag[tid] + grmag[tid + i];
    }
  }

  __syncthreads();

  _out[(bidx.z*griddim.y + bidx.y)*griddim.x + bidx.x] = grmag[0];

}



//
// For 3D
//
__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2F(float* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance)
{
  uint idx, idy, idz, tid;

  idx = blockIdx.x;
  idz = blockIdx.y / griddim.y;
  idy = blockIdx.y - idz*griddim.y;

  idx = idx*blockDim.x + threadIdx.x;
  idy = idy*blockDim.y + threadIdx.y;
  idz = idz*blockDim.z + threadIdx.z;
  
  __shared__ float shmem[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8
  __shared__ float grad_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient x
  __shared__ float grad_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient y
  __shared__ float grad_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient z
  __shared__ float cond_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance x
  __shared__ float cond_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance y
  __shared__ float cond_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance z
 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    dim3 cidx;
    cidx.x = threadIdx.x + 1;
    cidx.y = threadIdx.y + 1;
    cidx.z = threadIdx.z + 1;

    //
    // load shared memory
    //
    shmem[cidx.x][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + idx];

    // loading neighbors : Neumann boundary condition
    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem[0][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem[blockDim.x + 1][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,0) - 1;
      shmem[cidx.x][0][cidx.z] = _in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem[cidx.x][blockDim.y + 1][cidx.z] = _in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,0) - 1;
      shmem[cidx.x][cidx.y][0] = _in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem[cidx.x][cidx.y][blockDim.z + 1] = _in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }
       
    __syncthreads();

    float ctr = shmem[cidx.x][cidx.y][cidx.z];

    // compute gradient 
    grad_x[cidx.x][cidx.y][cidx.z] = shmem[cidx.x + 1][cidx.y][cidx.z] - ctr;
    grad_y[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y + 1][cidx.z] - ctr;
    grad_z[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y][cidx.z + 1] - ctr;
    
    if(threadIdx.x == 0) grad_x[cidx.x - 1][cidx.y][cidx.z] = ctr - shmem[cidx.x - 1][cidx.y][cidx.z];
    if(threadIdx.y == 0) grad_y[cidx.x][cidx.y - 1][cidx.z] = ctr - shmem[cidx.x][cidx.y - 1][cidx.z];
    if(threadIdx.z == 0) grad_z[cidx.x][cidx.y][cidx.z - 1] = ctr - shmem[cidx.x][cidx.y][cidx.z - 1];

    __syncthreads();

    // compute conductance
    float cond;
    
    cond = grad_x[cidx.x][cidx.y][cidx.z]/conductance;
    cond_x[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_y[cidx.x][cidx.y][cidx.z]/conductance;
    cond_y[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_z[cidx.x][cidx.y][cidx.z]/conductance;
    cond_z[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    if(threadIdx.x == 0) 
    {
      cond = grad_x[cidx.x - 1][cidx.y][cidx.z]/conductance;
      cond_x[cidx.x - 1][cidx.y][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.y == 0) 
    {
      cond = grad_y[cidx.x][cidx.y - 1][cidx.z]/conductance;
      cond_y[cidx.x][cidx.y - 1][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.z == 0)
    {
      cond = grad_z[cidx.x][cidx.y][cidx.z - 1]/conductance;
      cond_z[cidx.x][cidx.y][cidx.z - 1] = 1.0f/exp(cond*cond);
    }

    __syncthreads();  
    
    // sum up neighbors and multiply with timestep
    _out[((idz)*imagedim.y + (idy))*imagedim.x + idx] = shmem[cidx.x][cidx.y][cidx.z] +
                                                        (timestep)*(grad_x[cidx.x][cidx.y][cidx.z]*cond_x[cidx.x][cidx.y][cidx.z] -
                                                                         grad_x[cidx.x - 1][cidx.y][cidx.z]*cond_x[cidx.x - 1][cidx.y][cidx.z] +
                                                                         grad_y[cidx.x][cidx.y][cidx.z]*cond_y[cidx.x][cidx.y][cidx.z] -
                                                                         grad_y[cidx.x][cidx.y - 1][cidx.z]*cond_y[cidx.x][cidx.y - 1][cidx.z] +
                                                                         grad_z[cidx.x][cidx.y][cidx.z]*cond_z[cidx.x][cidx.y][cidx.z] -
                                                                         grad_z[cidx.x][cidx.y][cidx.z - 1]*cond_z[cidx.x][cidx.y][cidx.z - 1]);
                
  }
}


__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_U2F(uchar* _in, float* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance)
{
  uint idx, idy, idz, tid;

  idx = blockIdx.x;
  idz = blockIdx.y / griddim.y;
  idy = blockIdx.y - idz*griddim.y;

  idx = idx*blockDim.x + threadIdx.x;
  idy = idy*blockDim.y + threadIdx.y;
  idz = idz*blockDim.z + threadIdx.z;
  
  __shared__ float shmem[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8
  __shared__ float grad_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient x
  __shared__ float grad_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient y
  __shared__ float grad_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient z
  __shared__ float cond_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance x
  __shared__ float cond_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance y
  __shared__ float cond_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance z
 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    dim3 cidx;
    cidx.x = threadIdx.x + 1;
    cidx.y = threadIdx.y + 1;
    cidx.z = threadIdx.z + 1;

    //
    // load shared memory
    //
    shmem[cidx.x][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + idx];

    // loading neighbors : Neumann boundary condition
    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem[0][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem[blockDim.x + 1][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,0) - 1;
      shmem[cidx.x][0][cidx.z] = (float)_in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem[cidx.x][blockDim.y + 1][cidx.z] = (float)_in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,0) - 1;
      shmem[cidx.x][cidx.y][0] = (float)_in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem[cidx.x][cidx.y][blockDim.z + 1] = (float)_in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }
       
    __syncthreads();

    float ctr = shmem[cidx.x][cidx.y][cidx.z];

    // compute gradient 
    grad_x[cidx.x][cidx.y][cidx.z] = shmem[cidx.x + 1][cidx.y][cidx.z] - ctr;
    grad_y[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y + 1][cidx.z] - ctr;
    grad_z[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y][cidx.z + 1] - ctr;
    
    if(threadIdx.x == 0) grad_x[cidx.x - 1][cidx.y][cidx.z] = ctr - shmem[cidx.x - 1][cidx.y][cidx.z];
    if(threadIdx.y == 0) grad_y[cidx.x][cidx.y - 1][cidx.z] = ctr - shmem[cidx.x][cidx.y - 1][cidx.z];
    if(threadIdx.z == 0) grad_z[cidx.x][cidx.y][cidx.z - 1] = ctr - shmem[cidx.x][cidx.y][cidx.z - 1];

    __syncthreads();

    // compute conductance
    float cond;
    
    cond = grad_x[cidx.x][cidx.y][cidx.z]/conductance;
    cond_x[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_y[cidx.x][cidx.y][cidx.z]/conductance;
    cond_y[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_z[cidx.x][cidx.y][cidx.z]/conductance;
    cond_z[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    if(threadIdx.x == 0) 
    {
      cond = grad_x[cidx.x - 1][cidx.y][cidx.z]/conductance;
      cond_x[cidx.x - 1][cidx.y][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.y == 0) 
    {
      cond = grad_y[cidx.x][cidx.y - 1][cidx.z]/conductance;
      cond_y[cidx.x][cidx.y - 1][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.z == 0)
    {
      cond = grad_z[cidx.x][cidx.y][cidx.z - 1]/conductance;
      cond_z[cidx.x][cidx.y][cidx.z - 1] = 1.0f/exp(cond*cond);
    }

    __syncthreads();  
    
    // sum up neighbors and multiply with timestep
    _out[((idz)*imagedim.y + (idy))*imagedim.x + idx] = shmem[cidx.x][cidx.y][cidx.z] +
                                                        (timestep)*(grad_x[cidx.x][cidx.y][cidx.z]*cond_x[cidx.x][cidx.y][cidx.z] -
                                                                    grad_x[cidx.x - 1][cidx.y][cidx.z]*cond_x[cidx.x - 1][cidx.y][cidx.z] +
                                                                    grad_y[cidx.x][cidx.y][cidx.z]*cond_y[cidx.x][cidx.y][cidx.z] -
                                                                    grad_y[cidx.x][cidx.y - 1][cidx.z]*cond_y[cidx.x][cidx.y - 1][cidx.z] +
                                                                    grad_z[cidx.x][cidx.y][cidx.z]*cond_z[cidx.x][cidx.y][cidx.z] -
                                                                    grad_z[cidx.x][cidx.y][cidx.z - 1]*cond_z[cidx.x][cidx.y][cidx.z - 1]);
                
  }
}



__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_F2U(float* _in, uchar* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance)
{
  uint idx, idy, idz, tid;

  idx = blockIdx.x;
  idz = blockIdx.y / griddim.y;
  idy = blockIdx.y - idz*griddim.y;

  idx = idx*blockDim.x + threadIdx.x;
  idy = idy*blockDim.y + threadIdx.y;
  idz = idz*blockDim.z + threadIdx.z;
  
  __shared__ float shmem[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8
  __shared__ float grad_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient x
  __shared__ float grad_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient y
  __shared__ float grad_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient z
  __shared__ float cond_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance x
  __shared__ float cond_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance y
  __shared__ float cond_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance z
 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    dim3 cidx;
    cidx.x = threadIdx.x + 1;
    cidx.y = threadIdx.y + 1;
    cidx.z = threadIdx.z + 1;

    //
    // load shared memory
    //
    shmem[cidx.x][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + idx];

    // loading neighbors : Neumann boundary condition
    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem[0][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem[blockDim.x + 1][cidx.y][cidx.z] = _in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,0) - 1;
      shmem[cidx.x][0][cidx.z] = _in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem[cidx.x][blockDim.y + 1][cidx.z] = _in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,0) - 1;
      shmem[cidx.x][cidx.y][0] = _in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem[cidx.x][cidx.y][blockDim.z + 1] = _in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }
       
    __syncthreads();

    float ctr = shmem[cidx.x][cidx.y][cidx.z];

    // compute gradient 
    grad_x[cidx.x][cidx.y][cidx.z] = shmem[cidx.x + 1][cidx.y][cidx.z] - ctr;
    grad_y[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y + 1][cidx.z] - ctr;
    grad_z[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y][cidx.z + 1] - ctr;
    
    if(threadIdx.x == 0) grad_x[cidx.x - 1][cidx.y][cidx.z] = ctr - shmem[cidx.x - 1][cidx.y][cidx.z];
    if(threadIdx.y == 0) grad_y[cidx.x][cidx.y - 1][cidx.z] = ctr - shmem[cidx.x][cidx.y - 1][cidx.z];
    if(threadIdx.z == 0) grad_z[cidx.x][cidx.y][cidx.z - 1] = ctr - shmem[cidx.x][cidx.y][cidx.z - 1];

    __syncthreads();

    // compute conductance
    float cond;
    
    cond = grad_x[cidx.x][cidx.y][cidx.z]/conductance;
    cond_x[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_y[cidx.x][cidx.y][cidx.z]/conductance;
    cond_y[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_z[cidx.x][cidx.y][cidx.z]/conductance;
    cond_z[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    if(threadIdx.x == 0) 
    {
      cond = grad_x[cidx.x - 1][cidx.y][cidx.z]/conductance;
      cond_x[cidx.x - 1][cidx.y][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.y == 0) 
    {
      cond = grad_y[cidx.x][cidx.y - 1][cidx.z]/conductance;
      cond_y[cidx.x][cidx.y - 1][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.z == 0)
    {
      cond = grad_z[cidx.x][cidx.y][cidx.z - 1]/conductance;
      cond_z[cidx.x][cidx.y][cidx.z - 1] = 1.0f/exp(cond*cond);
    }

    __syncthreads();  
    
    // sum up neighbors and multiply with timestep
    _out[((idz)*imagedim.y + (idy))*imagedim.x + idx] = (uchar) ( shmem[cidx.x][cidx.y][cidx.z] +
                                                                  (timestep)*(grad_x[cidx.x][cidx.y][cidx.z]*cond_x[cidx.x][cidx.y][cidx.z] -
                                                                              grad_x[cidx.x - 1][cidx.y][cidx.z]*cond_x[cidx.x - 1][cidx.y][cidx.z] +
                                                                              grad_y[cidx.x][cidx.y][cidx.z]*cond_y[cidx.x][cidx.y][cidx.z] -
                                                                              grad_y[cidx.x][cidx.y - 1][cidx.z]*cond_y[cidx.x][cidx.y - 1][cidx.z] +
                                                                              grad_z[cidx.x][cidx.y][cidx.z]*cond_z[cidx.x][cidx.y][cidx.z] -
                                                                              grad_z[cidx.x][cidx.y][cidx.z - 1]*cond_z[cidx.x][cidx.y][cidx.z - 1]) );
                
  }
}



__global__ void kernel_GradientAnisotropicDiffusionFilter_NOSHAREDMEM_U2U(uchar* _in, uchar* _out, int dimension, dim3 imagedim, dim3 griddim, float timestep, float conductance)
{
  uint idx, idy, idz, tid;

  idx = blockIdx.x;
  idz = blockIdx.y / griddim.y;
  idy = blockIdx.y - idz*griddim.y;

  idx = idx*blockDim.x + threadIdx.x;
  idy = idy*blockDim.y + threadIdx.y;
  idz = idz*blockDim.z + threadIdx.z;
  
  __shared__ float shmem[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8
  __shared__ float grad_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient x
  __shared__ float grad_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient y
  __shared__ float grad_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // gradient z
  __shared__ float cond_x[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance x
  __shared__ float cond_y[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance y
  __shared__ float cond_z[BLOCK_DIM_X+1][BLOCK_DIM_Y+1][BLOCK_DIM_Z+1];  // conductance z
 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    dim3 cidx;
    cidx.x = threadIdx.x + 1;
    cidx.y = threadIdx.y + 1;
    cidx.z = threadIdx.z + 1;

    //
    // load shared memory
    //
    shmem[cidx.x][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + idx];

    // loading neighbors : Neumann boundary condition
    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem[0][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem[blockDim.x + 1][cidx.y][cidx.z] = (float)_in[((idz)*imagedim.y + (idy))*imagedim.x + tid];
    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,0) - 1;
      shmem[cidx.x][0][cidx.z] = (float)_in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem[cidx.x][blockDim.y + 1][cidx.z] = (float)_in[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,0) - 1;
      shmem[cidx.x][cidx.y][0] = (float)_in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem[cidx.x][cidx.y][blockDim.z + 1] = (float)_in[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }
       
    __syncthreads();

    float ctr = shmem[cidx.x][cidx.y][cidx.z];

    // compute gradient 
    grad_x[cidx.x][cidx.y][cidx.z] = shmem[cidx.x + 1][cidx.y][cidx.z] - ctr;
    grad_y[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y + 1][cidx.z] - ctr;
    grad_z[cidx.x][cidx.y][cidx.z] = shmem[cidx.x][cidx.y][cidx.z + 1] - ctr;
    
    if(threadIdx.x == 0) grad_x[cidx.x - 1][cidx.y][cidx.z] = ctr - shmem[cidx.x - 1][cidx.y][cidx.z];
    if(threadIdx.y == 0) grad_y[cidx.x][cidx.y - 1][cidx.z] = ctr - shmem[cidx.x][cidx.y - 1][cidx.z];
    if(threadIdx.z == 0) grad_z[cidx.x][cidx.y][cidx.z - 1] = ctr - shmem[cidx.x][cidx.y][cidx.z - 1];

    __syncthreads();

    // compute conductance
    float cond;
    
    cond = grad_x[cidx.x][cidx.y][cidx.z]/conductance;
    cond_x[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_y[cidx.x][cidx.y][cidx.z]/conductance;
    cond_y[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    cond = grad_z[cidx.x][cidx.y][cidx.z]/conductance;
    cond_z[cidx.x][cidx.y][cidx.z] = 1.0f/exp(cond*cond);

    if(threadIdx.x == 0) 
    {
      cond = grad_x[cidx.x - 1][cidx.y][cidx.z]/conductance;
      cond_x[cidx.x - 1][cidx.y][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.y == 0) 
    {
      cond = grad_y[cidx.x][cidx.y - 1][cidx.z]/conductance;
      cond_y[cidx.x][cidx.y - 1][cidx.z] = 1.0f/exp(cond*cond);
    }

    if(threadIdx.z == 0)
    {
      cond = grad_z[cidx.x][cidx.y][cidx.z - 1]/conductance;
      cond_z[cidx.x][cidx.y][cidx.z - 1] = 1.0f/exp(cond*cond);
    }

    __syncthreads();  
    
    // sum up neighbors and multiply with timestep
    _out[((idz)*imagedim.y + (idy))*imagedim.x + idx] = (uchar) ( shmem[cidx.x][cidx.y][cidx.z] +
                                                                  (timestep)*(grad_x[cidx.x][cidx.y][cidx.z]*cond_x[cidx.x][cidx.y][cidx.z] -
                                                                              grad_x[cidx.x - 1][cidx.y][cidx.z]*cond_x[cidx.x - 1][cidx.y][cidx.z] +
                                                                              grad_y[cidx.x][cidx.y][cidx.z]*cond_y[cidx.x][cidx.y][cidx.z] -
                                                                              grad_y[cidx.x][cidx.y - 1][cidx.z]*cond_y[cidx.x][cidx.y - 1][cidx.z] +
                                                                              grad_z[cidx.x][cidx.y][cidx.z]*cond_z[cidx.x][cidx.y][cidx.z] -
                                                                              grad_z[cidx.x][cidx.y][cidx.z - 1]*cond_z[cidx.x][cidx.y][cidx.z - 1]) );
                
  }
}









