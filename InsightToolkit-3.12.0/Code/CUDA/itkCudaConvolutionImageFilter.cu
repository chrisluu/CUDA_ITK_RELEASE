#include "itkCudaCommon.h"

//
// Reordering output global memory is required for 1D Convolution filter
// because too distanct global memory access causes a huge speed drop.
//
#define REORDER_GLOBAL_MEMORY

// shared memory
extern __shared__ float sharedmem[];

// constant memory that kernel can see
__constant__ float _kernel[MAX_KERNEL_SIZE];

__global__ void kernel_ConvolutionFilter_1D_F2F(float* _in, float* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);
__global__ void kernel_ConvolutionFilter_1D_U2F(uchar* _in, float* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);
__global__ void kernel_ConvolutionFilter_1D_F2U(float* _in, uchar* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);
__global__ void kernel_ConvolutionFilter_1D_U2U(uchar* _in, uchar* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);

__global__ void kernel_ConvolutionFilter_1D_F2F_Order(float* _in, float* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);
__global__ void kernel_ConvolutionFilter_1D_U2F_Order(uchar* _in, float* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);
__global__ void kernel_ConvolutionFilter_1D_F2U_Order(float* _in, uchar* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);
__global__ void kernel_ConvolutionFilter_1D_U2U_Order(uchar* _in, uchar* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread);



//
// Compute index for Global memory
//
__device__ uint get_global_index(int order, dim3 imdim, uint x, uint y, uint z)
{
  uint idx;

  if(order == 0)
  {
    idx = (z*(imdim.y) + y)*imdim.x + x;
  }
  else if(order == 1)
  {
    idx = (x*(imdim.z) + z)*imdim.y + y;  
  }
  else
  {
    idx = (y*(imdim.x) + x)*imdim.z + z;
  }
  
  return idx;
}

//
// Global memory order (indir/outdir)
//
//  0 (fast x) : x/y/z
//  1 (fast y) : y/z/x
//  2 (fast z) : z/x/y
//
// 3D block dimension is always block_size*1*1 (elongate along one axis)
// therefore, blockDim.x represents the block_size
//
__global__ void kernel_ConvolutionFilter_1D_F2F_Order(float* _in, float* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1) >> 1;
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(indir == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, min(max(0,idx+sid),imagedim.x-1), idy, idz);
        sharedmem[sid] = _in[gid];
      }
    }

    idx += (halfkernelsize + threadIdx.x);
  }
  else if(indir == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, min(max(0,idy+sid),imagedim.y-1), idz);
        sharedmem[sid] = _in[gid];
      }
    }

    idy += (halfkernelsize + threadIdx.x);
  }
  else // z direction
  {    
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
   
    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, idy, min(max(0,idz+sid),imagedim.z-1));
        sharedmem[sid] = _in[gid];
      }
    }

    idz += (halfkernelsize + threadIdx.x);
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
     
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = get_global_index(outdir, imagedim, idx, idy, idz);
    _out[outidx] = _sum;
  }
}


__global__ void kernel_ConvolutionFilter_1D_U2F_Order(uchar* _in, float* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1) >> 1;
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(indir == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, min(max(0,idx+sid),imagedim.x-1), idy, idz);
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idx += (halfkernelsize + threadIdx.x);
  }
  else if(indir == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, min(max(0,idy+sid),imagedim.y-1), idz);
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idy += (halfkernelsize + threadIdx.x);
  }
  else // z direction
  {    
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
   
    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, idy, min(max(0,idz+sid),imagedim.z-1));
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idz += (halfkernelsize + threadIdx.x);
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
     
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = get_global_index(outdir, imagedim, idx, idy, idz);
    _out[outidx] = _sum;
  }
}


__global__ void kernel_ConvolutionFilter_1D_F2U_Order(float* _in, uchar* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1) >> 1;
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(indir == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, min(max(0,idx+sid),imagedim.x-1), idy, idz);
        sharedmem[sid] = _in[gid];
      }
    }

    idx += (halfkernelsize + threadIdx.x);
  }
  else if(indir == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, min(max(0,idy+sid),imagedim.y-1), idz);
        sharedmem[sid] = _in[gid];
      }
    }

    idy += (halfkernelsize + threadIdx.x);
  }
  else // z direction
  {    
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
   
    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, idy, min(max(0,idz+sid),imagedim.z-1));
        sharedmem[sid] = _in[gid];
      }
    }

    idz += (halfkernelsize + threadIdx.x);
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
     
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = get_global_index(outdir, imagedim, idx, idy, idz);
    _out[outidx] = (uchar)_sum;
  }
}

__global__ void kernel_ConvolutionFilter_1D_U2U_Order(uchar* _in, uchar* _out, int indir, int outdir, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1) >> 1;
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(indir == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, min(max(0,idx+sid),imagedim.x-1), idy, idz);
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idx += (halfkernelsize + threadIdx.x);
  }
  else if(indir == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, min(max(0,idy+sid),imagedim.y-1), idz);
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idy += (halfkernelsize + threadIdx.x);
  }
  else // z direction
  {    
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
   
    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = get_global_index(indir, imagedim, idx, idy, min(max(0,idz+sid),imagedim.z-1));
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idz += (halfkernelsize + threadIdx.x);
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
     
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = get_global_index(outdir, imagedim, idx, idy, idz);
    _out[outidx] = (uchar)_sum;
  }
}



//
// Version 1.1 
//
__global__ void kernel_ConvolutionFilter_1D_F2F(float* _in, float* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1) >> 1;/*/2;*/
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(direction == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = (((uint)idz)*imagedim.y + ((uint)idy))*imagedim.x + 
                   min(max(0,idx+(int)sid),imagedim.x-1);

#ifdef __DEVICE_EMULATION__
        assert(sharedMemSize > 0);
        assert(sid < sharedMemSize && sid >= 0);
        assert(gid < imagedim.x*imagedim.y*imagedim.z && gid >= 0);
#endif

        sharedmem[sid] = _in[gid];
      }
    }

    idx += (halfkernelsize + threadIdx.x);
  }
  else if(direction == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = (((uint)idz)*imagedim.y + min(max(0,idy+sid),imagedim.y-1))*imagedim.x + 
                   (uint)idx;
       
        sharedmem[sid] = _in[gid];
      }
    }

    idy += (halfkernelsize + threadIdx.x);
  }
  else // z direction
  {    
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
    
    uint gid = idy*imagedim.x + idx;                   

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      int sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint nid = gid + min(max(0,idz+sid),imagedim.z-1)*imagedim.x*imagedim.y;
        sharedmem[sid] = _in[nid];
      }
    }

    idz += (halfkernelsize + threadIdx.x);
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
     
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = (((uint)idz)*imagedim.y + ((uint)idy))*imagedim.x + (uint)idx;

#ifdef __DEVICE_EMULATION__       
        assert(outidx < imagedim.x*imagedim.y*imagedim.z && outidx >= 0);
#endif

    _out[outidx] = _sum;
  }
}


__global__ void kernel_ConvolutionFilter_1D_F2U(float* _in, uchar* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1)/2;
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(direction == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = ((uint)idz)*imagedim.x*imagedim.y + 
                   ((uint)idy)*imagedim.x + 
                   min(max(0,idx+(int)sid),imagedim.x-1);

        sharedmem[sid] = _in[gid];
      }
    }

    idx += halfkernelsize + threadIdx.x;
  }
  else if(direction == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = ((uint)idz)*imagedim.x*imagedim.y + 
                   min(max(0,idy+(int)sid),imagedim.y-1)*imagedim.x + 
                   (uint)idx;
 
         sharedmem[sid] = _in[gid];
      }
    }

    idy += halfkernelsize + threadIdx.x;
  }
  else // z direction
  {
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
    
    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = min(max(0,idz+(int)sid),imagedim.z-1)*imagedim.x*imagedim.y + 
                   ((uint)idy)*imagedim.x + 
                   (uint)idx;
        
        sharedmem[sid] = _in[gid];
      }
    }

    idz += halfkernelsize + threadIdx.x;
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
      
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = ((uint)idz)*imagedim.x*imagedim.y + ((uint)idy)*imagedim.x + (uint)idx;

    _out[outidx] = (uchar)_sum;
  }
}



__global__ void kernel_ConvolutionFilter_1D_U2F(uchar* _in, float* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1)/2;
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(direction == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = ((uint)idz)*imagedim.x*imagedim.y + 
                   ((uint)idy)*imagedim.x + 
                   min(max(0,idx+(int)sid),imagedim.x-1);

        sharedmem[sid] = (float)_in[gid];
      }
    }

    idx += halfkernelsize + threadIdx.x;
  }
  else if(direction == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = ((uint)idz)*imagedim.x*imagedim.y + 
                   min(max(0,idy+(int)sid),imagedim.y-1)*imagedim.x + 
                   (uint)idx;
 
         sharedmem[sid] = (float)_in[gid];
      }
    }

    idy += halfkernelsize + threadIdx.x;
  }
  else // z direction
  {
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
    
    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = min(max(0,idz+(int)sid),imagedim.z-1)*imagedim.x*imagedim.y + 
                   ((uint)idy)*imagedim.x + 
                   (uint)idx;
        
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idz += halfkernelsize + threadIdx.x;
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
      
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = ((uint)idz)*imagedim.x*imagedim.y + ((uint)idy)*imagedim.x + (uint)idx;

    _out[outidx] = _sum;
  }
}


__global__ void kernel_ConvolutionFilter_1D_U2U(uchar* _in, uchar* _out, int direction, dim3 imagedim, dim3 griddim, uint kernelsize, int nUpdatePerThread)
{
  // 
  // 1. initialize shared memory
  //
  uint bidx, bidy, bidz, sharedMemSize;
  int idx, idy, idz, halfkernelsize;

  halfkernelsize = (kernelsize - 1)/2;
  sharedMemSize = blockDim.x + kernelsize - 1;

  if(direction == 0) // x direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)(__mul24(bidx,blockDim.x)) - halfkernelsize;
    idy = (int)bidy;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      
      if(sid < sharedMemSize)
      { 
        uint gid = ((uint)idz)*imagedim.x*imagedim.y + 
                   ((uint)idy)*imagedim.x + 
                   min(max(0,idx+(int)sid),imagedim.x-1);

        sharedmem[sid] = (float)_in[gid];
      }
    }

    idx += halfkernelsize + threadIdx.x;
  }
  else if(direction == 1) // y direction
  {
    bidy = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidy*griddim.x;
    bidz = blockIdx.y;

    idx = (int)bidx;
    idy = (int)(__mul24(bidy,blockDim.x)) - halfkernelsize;
    idz = (int)bidz;

    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = ((uint)idz)*imagedim.x*imagedim.y + 
                   min(max(0,idy+(int)sid),imagedim.y-1)*imagedim.x + 
                   (uint)idx;
 
         sharedmem[sid] = (float)_in[gid];
      }
    }

    idy += halfkernelsize + threadIdx.x;
  }
  else // z direction
  {
    bidz = blockIdx.x / griddim.x;
    bidx = blockIdx.x - bidz*griddim.x;
    bidy = blockIdx.y;

    idx = (int)bidx;
    idy = (int)bidy;
    idz = (int)(__mul24(bidz,blockDim.x)) - halfkernelsize;
    
    for(uint i=0; i<nUpdatePerThread; i++)
    {
      uint sid = threadIdx.x + i*blockDim.x; // index in shared memory      
      if(sid < sharedMemSize)
      { 
        uint gid = min(max(0,idz+(int)sid),imagedim.z-1)*imagedim.x*imagedim.y + 
                   ((uint)idy)*imagedim.x + 
                   (uint)idx;
        
        sharedmem[sid] = (float)_in[gid];
      }
    }

    idz += halfkernelsize + threadIdx.x;
  }
 
  __syncthreads();


  // 
  // 2. apply kernel
  // 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
      
    for(uint i=threadIdx.x; i<threadIdx.x+kernelsize; i++)
    {
      _sum += sharedmem[i]*_kernel[i-threadIdx.x];    
    }

    // store the result
    uint outidx = ((uint)idz)*imagedim.x*imagedim.y + ((uint)idy)*imagedim.x + (uint)idx;

    _out[outidx] = (uchar)_sum;
  }
}



// 1D convolution 
extern "C"
void cudaConvolutionFilter1DKernelWrapper(void *_in, void *_out, float *kernel,
                                          FILTERTYPE type, int indir, int outdir, int kernelsize, 
                                          uint image_width, uint image_height, uint image_depth)
{
	// 8 : max block width = 256
	// 7 : max block width = 128
	int maxBlockWidthPowSize = 7; // max block width = 2^maxBlockWidthPowSize

	uint blockWidth, blockHeight, blockDepth;
	uint nBlockX, nBlockY, nBlockZ;
	uint sharedmemsize;

	// copy kernel to constant memory
	cudaMemcpyToSymbol((const char*)_kernel, kernel, sizeof(float)*kernelsize);

	bool reorder = false;
	if(indir != outdir) reorder = true;

	switch(reorder)
	{
	case true:
		{
			// compute how many blocks are needed
			dim3 griddim; // actual grid dimension
			if(indir == 0) // x
			{
				uint bestfitsize = log2(image_width);
				blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
				blockHeight = 1;
				blockDepth  = 1;
				sharedmemsize = blockWidth + kernelsize - 1;

				nBlockX = ceil((float)image_width / (float)blockWidth);
				nBlockY = image_height;
				nBlockZ = image_depth;
				griddim = dim3(nBlockX,nBlockY,nBlockZ);
			}
			else if(indir == 1)
			{
				uint bestfitsize = log2(image_height);
				blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
				blockHeight = 1;
				blockDepth  = 1;
				sharedmemsize = blockWidth + kernelsize - 1;

				nBlockX = ceil((float)image_height / (float)blockWidth);
				nBlockY = image_width;
				nBlockZ = image_depth;
				griddim = dim3(nBlockY,nBlockX,nBlockZ);
			}
			else
			{
				uint bestfitsize = log2(image_depth);
				blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
				blockHeight = 1;
				blockDepth  = 1;
				sharedmemsize = blockWidth + kernelsize - 1;

				nBlockX = ceil((float)image_depth / (float)blockWidth);
				nBlockY = image_width;
				nBlockZ = image_height;
				griddim = dim3(nBlockY,nBlockZ,nBlockX);
			}

			dim3 dimGrid(nBlockX*nBlockY,nBlockZ); // 3D grid is not supported on G80
			dim3 dimBlock(blockWidth, blockHeight, blockDepth);
			dim3 imagedim(image_width,image_height,image_depth);

			uint updateperthread = ceil((float)(sharedmemsize)/(float)(blockWidth*blockHeight*blockDepth));

			// kernel call
			if(type == FLOAT_TO_FLOAT)
			{
				kernel_ConvolutionFilter_1D_F2F_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (float*)_out, indir, outdir, imagedim, griddim, kernelsize, updateperthread);
			}
			else if(type == FLOAT_TO_UCHAR)
			{
				kernel_ConvolutionFilter_1D_F2U_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (uchar*)_out, indir, outdir,  imagedim, griddim, kernelsize, updateperthread);
			}
			else if(type == UCHAR_TO_FLOAT)
			{
				kernel_ConvolutionFilter_1D_U2F_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (float*)_out, indir, outdir, imagedim, griddim, kernelsize, updateperthread);
			}
			else if(type == UCHAR_TO_UCHAR)
			{
				kernel_ConvolutionFilter_1D_U2U_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (uchar*)_out, indir, outdir, imagedim, griddim, kernelsize, updateperthread);
			}
			else
			{
				printf("Unsupported pixel type\n");
				exit(0);
			}
			CUT_CHECK_ERROR("Kernel execution failed"); 
		}
		break;

	case false:
		{

			int direction = indir;

			// compute how many blocks are needed
			dim3 griddim; // actual grid dimension
			if(direction == 0) // x
			{
				uint bestfitsize = log2(image_width);
				blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
				blockHeight = 1;
				blockDepth  = 1;
				sharedmemsize = blockWidth + kernelsize - 1;

				nBlockX = ceil((float)image_width / (float)blockWidth);
				nBlockY = image_height;
				nBlockZ = image_depth;
				griddim = dim3(nBlockX,nBlockY,nBlockZ);
			}
			else if(direction == 1)
			{
				uint bestfitsize = log2(image_height);
				blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
				blockHeight = 1;
				blockDepth  = 1;
				sharedmemsize = blockWidth + kernelsize - 1;

				nBlockX = ceil((float)image_height / (float)blockWidth);
				nBlockY = image_width;
				nBlockZ = image_depth;
				griddim = dim3(nBlockY,nBlockX,nBlockZ);
			}
			else
			{
				uint bestfitsize = log2(image_depth);
				blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
				blockHeight = 1;
				blockDepth  = 1;
				sharedmemsize = blockWidth + kernelsize - 1;

				nBlockX = ceil((float)image_depth / (float)blockWidth); // # of blocks along z
				nBlockY = image_width;  // # of blocks along x
				nBlockZ = image_height; // # of blocks along y 
				griddim = dim3(nBlockY,nBlockZ,nBlockX);
			}

			dim3 dimGrid(nBlockX*nBlockY,nBlockZ); // 3D grid is not supported on G80
			dim3 dimBlock(blockWidth, blockHeight, blockDepth);
			dim3 imagedim(image_width,image_height,image_depth);

			uint updateperthread = ceil((float)(sharedmemsize)/(float)(blockWidth*blockHeight*blockDepth));

			// kernel call
			if(type == FLOAT_TO_FLOAT)
			{
				kernel_ConvolutionFilter_1D_F2F<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (float*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
			}
			else if(type == FLOAT_TO_UCHAR)
			{
				kernel_ConvolutionFilter_1D_F2U<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (uchar*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
			}
			else if(type == UCHAR_TO_FLOAT)
			{
				kernel_ConvolutionFilter_1D_U2F<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (float*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
			}
			else if(type == UCHAR_TO_UCHAR)
			{
				kernel_ConvolutionFilter_1D_U2U<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (uchar*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
			}
			else
			{
				printf("Unsupported pixel type\n");
				exit(0);
			}
			CUT_CHECK_ERROR("Kernel execution failed"); 
		}
		break;

	default:
		break;
	}

}


/*
// 1D convolution 
extern "C"
void cudaConvolutionFilter1DKernelWrapper(void *_in, void *_out, float *kernel,
                                          FILTERTYPE type, int indir, int outdir, int kernelsize, 
                                          uint image_width, uint image_height, uint image_depth)
{
  // 8 : max block width = 256
  // 7 : max block width = 128
  int maxBlockWidthPowSize = 7; // max block width = 2^maxBlockWidthPowSize

  uint blockWidth, blockHeight, blockDepth;
  uint nBlockX, nBlockY, nBlockZ;
  uint sharedmemsize;
   
  // copy kernel to constant memory
  cudaMemcpyToSymbol((const char*)_kernel, kernel, sizeof(float)*kernelsize);
 
#ifdef REORDER_GLOBAL_MEMORY

  // compute how many blocks are needed
  dim3 griddim; // actual grid dimension
  if(indir == 0) // x
  {
    uint bestfitsize = log2(image_width);
    blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
    blockHeight = 1;
    blockDepth  = 1;
    sharedmemsize = blockWidth + kernelsize - 1;

    nBlockX = ceil((float)image_width / (float)blockWidth);
    nBlockY = image_height;
    nBlockZ = image_depth;
    griddim = dim3(nBlockX,nBlockY,nBlockZ);
  }
  else if(indir == 1)
  {
    uint bestfitsize = log2(image_height);
    blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
    blockHeight = 1;
    blockDepth  = 1;
    sharedmemsize = blockWidth + kernelsize - 1;

    nBlockX = ceil((float)image_height / (float)blockWidth);
    nBlockY = image_width;
    nBlockZ = image_depth;
    griddim = dim3(nBlockY,nBlockX,nBlockZ);
  }
  else
  {
    uint bestfitsize = log2(image_depth);
    blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
    blockHeight = 1;
    blockDepth  = 1;
    sharedmemsize = blockWidth + kernelsize - 1;

    nBlockX = ceil((float)image_depth / (float)blockWidth);
    nBlockY = image_width;
    nBlockZ = image_height;
    griddim = dim3(nBlockY,nBlockZ,nBlockX);
  }
 
  dim3 dimGrid(nBlockX*nBlockY,nBlockZ); // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight, blockDepth);
  dim3 imagedim(image_width,image_height,image_depth);
  
  uint updateperthread = ceil((float)(sharedmemsize)/(float)(blockWidth*blockHeight*blockDepth));

  // kernel call
  if(type == FLOAT_TO_FLOAT)
  {
    kernel_ConvolutionFilter_1D_F2F_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (float*)_out, indir, outdir, imagedim, griddim, kernelsize, updateperthread);
  }
  else if(type == FLOAT_TO_UCHAR)
  {
    kernel_ConvolutionFilter_1D_F2U_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (uchar*)_out, indir, outdir,  imagedim, griddim, kernelsize, updateperthread);
  }
  else if(type == UCHAR_TO_FLOAT)
  {
    kernel_ConvolutionFilter_1D_U2F_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (float*)_out, indir, outdir, imagedim, griddim, kernelsize, updateperthread);
  }
  else if(type == UCHAR_TO_UCHAR)
  {
    kernel_ConvolutionFilter_1D_U2U_Order<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (uchar*)_out, indir, outdir, imagedim, griddim, kernelsize, updateperthread);
  }
  else
  {
    printf("Unsupported pixel type\n");
    exit(0);
  }
  CUT_CHECK_ERROR("Kernel execution failed"); 

#else

  int direction = indir;

  // compute how many blocks are needed
  dim3 griddim; // actual grid dimension
  if(direction == 0) // x
  {
    uint bestfitsize = log2(image_width);
    blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
    blockHeight = 1;
    blockDepth  = 1;
    sharedmemsize = blockWidth + kernelsize - 1;

    nBlockX = ceil((float)image_width / (float)blockWidth);
    nBlockY = image_height;
    nBlockZ = image_depth;
    griddim = dim3(nBlockX,nBlockY,nBlockZ);
  }
  else if(direction == 1)
  {
    uint bestfitsize = log2(image_height);
    blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
    blockHeight = 1;
    blockDepth  = 1;
    sharedmemsize = blockWidth + kernelsize - 1;

    nBlockX = ceil((float)image_height / (float)blockWidth);
    nBlockY = image_width;
    nBlockZ = image_depth;
    griddim = dim3(nBlockY,nBlockX,nBlockZ);
  }
  else
  {
    uint bestfitsize = log2(image_depth);
    blockWidth  = 2 << (min(bestfitsize,maxBlockWidthPowSize)-1);
    blockHeight = 1;
    blockDepth  = 1;
    sharedmemsize = blockWidth + kernelsize - 1;

    nBlockX = ceil((float)image_depth / (float)blockWidth); // # of blocks along z
    nBlockY = image_width;  // # of blocks along x
    nBlockZ = image_height; // # of blocks along y 
    griddim = dim3(nBlockY,nBlockZ,nBlockX);
  }
 
  dim3 dimGrid(nBlockX*nBlockY,nBlockZ); // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight, blockDepth);
  dim3 imagedim(image_width,image_height,image_depth);
  
  uint updateperthread = ceil((float)(sharedmemsize)/(float)(blockWidth*blockHeight*blockDepth));
  
  // kernel call
  if(type == FLOAT_TO_FLOAT)
  {
    kernel_ConvolutionFilter_1D_F2F<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (float*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
  }
  else if(type == FLOAT_TO_UCHAR)
  {
    kernel_ConvolutionFilter_1D_F2U<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((float*)_in, (uchar*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
  }
  else if(type == UCHAR_TO_FLOAT)
  {
    kernel_ConvolutionFilter_1D_U2F<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (float*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
  }
  else if(type == UCHAR_TO_UCHAR)
  {
    kernel_ConvolutionFilter_1D_U2U<<<dimGrid,dimBlock,sharedmemsize*sizeof(float)>>>((uchar*)_in, (uchar*)_out, direction, imagedim, griddim, kernelsize, updateperthread);
  }
  else
  {
    printf("Unsupported pixel type\n");
    exit(0);
  }
  CUT_CHECK_ERROR("Kernel execution failed"); 

#endif

}
*/