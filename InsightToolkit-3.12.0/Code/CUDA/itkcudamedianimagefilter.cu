#include "itkCudaCommon.h"

extern __shared__ unsigned char sharedmem[];

__global__ void kernel_MedianFilter_U2U(uchar* _in, uchar* _out, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float halfneighborsize);


// extern function to call kernel 
extern "C"
void cudaMedianImageFilterKernelWrapper(void *_in, void *_out,
                                        FILTERTYPE type, int dimension,
                                        uint image_width, uint image_height, uint image_depth,
                                        uint kernel_width, uint kernel_height, uint kernel_depth)
{
  int blockWidth, blockHeight, blockDepth;
  int nBlockX, nBlockY, nBlockZ;

  // Setting block size
  if(dimension == 2) 
  {
    blockWidth  = 16;
    blockHeight = 16;
    blockDepth  = 1;

    image_depth = 1;
    kernel_depth = 0;
  }
  else
  {
    blockWidth  = 8;
    blockHeight = 8;
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
  dim3 kerneldim(kernel_width,kernel_height,kernel_depth);
  dim3 sharedmemdim((kernel_width*2)+blockWidth,(kernel_height*2)+blockHeight,(kernel_depth*2)+blockDepth);
  uint updateperthread = ceil((float)(sharedmemdim.x*sharedmemdim.y*sharedmemdim.z)/(float)(blockWidth*blockHeight*blockDepth));
  uint neighborsize = (2*kernel_width+1)*(2*kernel_height+1)*(2*kernel_depth+1);
 
  if(type == UCHAR_TO_UCHAR)
  { 
    uint sharedmemsize = sizeof(uchar)*sharedmemdim.x*sharedmemdim.y*sharedmemdim.z;
    kernel_MedianFilter_U2U<<<dimGrid,dimBlock,sharedmemsize>>>((uchar*)_in, (uchar*)_out, imagedim, griddim, kerneldim, sharedmemdim, updateperthread, ((float)neighborsize)/2.0f);
  }
  else
  {
    printf("Unsupported pixel type\n");
    exit(0);
  }
  CUT_CHECK_ERROR("Kernel execution failed"); 
}


__global__ void kernel_MedianFilter_U2U(uchar *_in, uchar *_out, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float halfneighborsize)
{
  //
  // ver 1.1 : use shared memory
  // 
  uint bidx = blockIdx.x;
  uint bidy = blockIdx.y%griddim.y;
  uint bidz = (uint)((blockIdx.y)/griddim.y);

  // global index for block endpoint
  uint beidx = __mul24(bidx,blockDim.x);
  uint beidy = __mul24(bidy,blockDim.y);
  uint beidz = __mul24(bidz,blockDim.z);

  uint tid = __mul24(threadIdx.z,__mul24(blockDim.x,blockDim.y)) + 
             __mul24(threadIdx.y,blockDim.x) + threadIdx.x;

#ifdef __DEVICE_EMULATION__
  printf("tid : %d\n", tid); 
#endif

  // update shared memory
  uint nthreads = blockDim.x*blockDim.y*blockDim.z;
  uint sharedMemSize = sharedmemdim.x * sharedmemdim.y * sharedmemdim.z;
  for(uint i=0; i<nUpdatePerThread; i++)
  {
    uint sid = tid + i*nthreads; // index in shared memory
    
    if(sid < sharedMemSize)
    { 
      // global x/y/z index in volume
      int gidx, gidy, gidz;
      uint sidx, sidy, sidz, tid;

      sidz = sid / (sharedmemdim.x*sharedmemdim.y);
      tid  = sid - sidz*(sharedmemdim.x*sharedmemdim.y);
      sidy = tid / (sharedmemdim.x);
      sidx = tid - sidy*(sharedmemdim.x);
     
      gidx = (int)sidx - (int)kerneldim.x + (int)beidx;
      gidy = (int)sidy - (int)kerneldim.y + (int)beidy;
      gidz = (int)sidz - (int)kerneldim.z + (int)beidz; 
     
      // Neumann boundary condition
      uint cx = (uint) min(max(0,gidx),imagedim.x-1);
      uint cy = (uint) min(max(0,gidy),imagedim.y-1);
      uint cz = (uint) min(max(0,gidz),imagedim.z-1);

      uint gid = cz*imagedim.x*imagedim.y + cy*imagedim.x + cx;
     
      sharedmem[sid] = _in[gid];
    }
  }

  __syncthreads();


  // global index of the current voxel in the input volume
  uint idx = beidx + threadIdx.x;
  uint idy = beidy + threadIdx.y;
  uint idz = beidz + threadIdx.z;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {   
    /*
    unsigned short hist[256]; // unsigned short

    // modified method
    for(int i=0; i<256; i++) hist[i] = 0;
    
    for(uint k=threadIdx.z; k<=threadIdx.z+2*kerneldim.z; k++)
    {
      for(uint j=threadIdx.y; j<=threadIdx.y+2*kerneldim.y; j++)
      {
        for(uint i=threadIdx.x; i<=threadIdx.x+2*kerneldim.x; i++)
        {
          int val = (int)sharedmem[(k*sharedmemdim.y + j)*sharedmemdim.x + i];
          hist[val] += 1;
        }
      }
    }

    uint count = 0;
    int pivot = -1;
    for(int i=0; i<256; i++)
    {
      count += (uint) hist[i];
      if(count >= halfneighborsize && pivot < 0)
      {
        pivot = i;        
      }
    }
    */
   

    // Viola's method
    uint minval = 0;
    uint maxval = 255;
    float pivot = (minval + maxval)/2.0f;

    for(int i=0; i<8; i++) // 8 bit pixel value
    {
      int count = 0;
     
      for(uint k=threadIdx.z; k<=threadIdx.z+2*kerneldim.z; k++)
      {
        for(uint j=threadIdx.y; j<=threadIdx.y+2*kerneldim.y; j++)
        {
          for(uint i=threadIdx.x; i<=threadIdx.x+2*kerneldim.x; i++)
          {
            int val = sharedmem[(k*sharedmemdim.y + j)*sharedmemdim.x + i];
            if(val > pivot) count++;
          }
        }
      }   

      if(count < halfneighborsize)
      {
        maxval = floor(pivot);      
      }
      else
      {
        minval = ceil(pivot);
      }

      pivot = (minval + maxval)/2.0f;
    }

    // store the result
    _out[(idz*imagedim.y + idy)*imagedim.x + idx] = (uchar)floor(pivot);
  }

}
