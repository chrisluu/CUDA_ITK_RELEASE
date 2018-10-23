#include "itkCudaCommon.h"

extern __shared__ float sharedmem[];

__global__ void kernel_MeanFilter_F2F_OFC(float* _in, float* _out, dim3 minoffset, dim3 maxoffset, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize);
__global__ void kernel_MeanFilter_U2U_OFC(uchar* _in, uchar* _out, dim3 minoffset, dim3 maxoffset, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize);

__global__ void kernel_MeanFilter_F2F(float* _in, float* _out, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize);
__global__ void kernel_MeanFilter_U2U(uchar* _in, uchar* _out, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize);


//
// Out-Of-Core version
//
extern "C"
void cudaMeanImageFilterKernelWrapperOFC(void *_in, void *_out,
                                         FILTERTYPE type, int dimension,
                                         uint image_width, uint image_height, uint image_depth,
                                         uint kernel_width, uint kernel_height, uint kernel_depth,
                                         uint minoffset_x, uint minoffset_y, uint minoffset_z,
                                         uint maxoffset_x, uint maxoffset_y, uint maxoffset_z)
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
    blockWidth  = 8; //4;//
    blockHeight = 8; //4;//
    blockDepth  = 4;     
  }

  // compute how many blocks are needed
  nBlockX = ceil((float)image_width / (float)blockWidth);
  nBlockY = ceil((float)image_height / (float)blockHeight);
  nBlockZ = ceil((float)image_depth / (float)blockDepth);
 
  dim3 dimGrid(nBlockX,nBlockY*nBlockZ); // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight, blockDepth);
  dim3 minoffset(minoffset_x, minoffset_y, minoffset_z);
  dim3 maxoffset(maxoffset_x, maxoffset_y, maxoffset_z); 
  dim3 imagedim(image_width,image_height,image_depth);
  dim3 griddim(nBlockX,nBlockY,nBlockZ);
  dim3 kerneldim(kernel_width,kernel_height,kernel_depth);
  dim3 sharedmemdim((kernel_width*2)+blockWidth,(kernel_height*2)+blockHeight,(kernel_depth*2)+blockDepth);
  uint sharedmemsize = sizeof(float)*sharedmemdim.x*sharedmemdim.y*sharedmemdim.z;
  uint updateperthread = ceil((float)(sharedmemdim.x*sharedmemdim.y*sharedmemdim.z)/(float)(blockWidth*blockHeight*blockDepth));
  float neighborsize = (2*kernel_width+1)*(2*kernel_height+1)*(2*kernel_depth+1);
 
  if(type == FLOAT_TO_FLOAT)
  {
    // kernel call
    kernel_MeanFilter_F2F_OFC<<<dimGrid,dimBlock,sharedmemsize>>>((float*)_in, (float*)_out, minoffset, maxoffset, imagedim, griddim, kerneldim, sharedmemdim, updateperthread, neighborsize);
  }
  else if(type == UCHAR_TO_UCHAR)
  {
    kernel_MeanFilter_U2U_OFC<<<dimGrid,dimBlock,sharedmemsize>>>((uchar*)_in, (uchar*)_out, minoffset, maxoffset, imagedim, griddim, kerneldim, sharedmemdim, updateperthread, neighborsize);
  }
  else
  {
    printf("Unsupported pixel type\n");
    exit(0);
  }
  CUT_CHECK_ERROR("Kernel execution failed"); 
}



// extern function to call kernel 
extern "C"
void cudaMeanImageFilterKernelWrapper(void *_in, void *_out,
                                      FILTERTYPE type, int dimension,
                                      uint image_width, uint image_height, uint image_depth,
                                      uint kernel_width, uint kernel_height, uint kernel_depth)
{

  cudaMeanImageFilterKernelWrapperOFC(_in, _out, type, dimension,
                                      image_width, image_height, image_depth,
                                      kernel_width, kernel_height, kernel_depth,
                                      0, 0, 0, 0, 0, 0);
}


/*
// extern function to call kernel 
extern "C"
void cudaMeanImageFilterKernelWrapper(void *_in, void *_out,
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
  uint sharedmemsize = sizeof(float)*sharedmemdim.x*sharedmemdim.y*sharedmemdim.z;
  uint updateperthread = ceil((float)(sharedmemdim.x*sharedmemdim.y*sharedmemdim.z)/(float)(blockWidth*blockHeight*blockDepth));
  float neighborsize = (2*kernel_width+1)*(2*kernel_height+1)*(2*kernel_depth+1);
 
  if(type == FLOAT_TO_FLOAT)
  {
    // kernel call
    kernel_MeanFilter_F2F<<<dimGrid,dimBlock,sharedmemsize>>>((float*)_in, (float*)_out, imagedim, griddim, kerneldim, sharedmemdim, updateperthread, neighborsize);
  }
  else if(type == UCHAR_TO_UCHAR)
  {
     kernel_MeanFilter_U2U<<<dimGrid,dimBlock,sharedmemsize>>>((uchar*)_in, (uchar*)_out, imagedim, griddim, kerneldim, sharedmemdim, updateperthread, neighborsize);
  }
  else
  {
    printf("Unsupported pixel type\n");
    exit(0);
  }
  CUT_CHECK_ERROR("Kernel execution failed"); 
}
*/

__global__ void kernel_MeanFilter_F2F_OFC(float *_in, float *_out, dim3 minoffset, dim3 maxoffset, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize)
{
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
  printf("tid : %d", tid); 
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

  // check range
  if( idx >= minoffset.x && idx < (imagedim.x - maxoffset.x) && 
      idy >= minoffset.y && idy < (imagedim.y - maxoffset.y) && 
      idz >= minoffset.z && idz < (imagedim.z - maxoffset.z) )
  {
    // sum neighbor pixel values
    float _sum = 0;
    for(uint k=threadIdx.z; k<=threadIdx.z+2*kerneldim.z; k++)
    {
      for(uint j=threadIdx.y; j<=threadIdx.y+2*kerneldim.y; j++)
      {
        for(uint i=threadIdx.x; i<=threadIdx.x+2*kerneldim.x; i++)
        {
          _sum += sharedmem[k*sharedmemdim.x*sharedmemdim.y + j*sharedmemdim.x + i];    
        }
      }
    }   

    // store the result
    _out[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = _sum/neighborsize;
  }
}

__global__ void kernel_MeanFilter_U2U_OFC(uchar *_in, uchar *_out, dim3 minoffset, dim3 maxoffset, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize)
{
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
  printf("tid : %d", tid); 
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
     
      sharedmem[sid] = (float) _in[gid];
    }
  }

  __syncthreads();


  // global index of the current voxel in the input volume
  uint idx = beidx + threadIdx.x;
  uint idy = beidy + threadIdx.y;
  uint idz = beidz + threadIdx.z;

  // check range
  if( idx >= minoffset.x && idx < (imagedim.x - maxoffset.x) && 
      idy >= minoffset.y && idy < (imagedim.y - maxoffset.y) && 
      idz >= minoffset.z && idz < (imagedim.z - maxoffset.z) )
  {
    // sum neighbor pixel values
    float _sum = 0;
    for(uint k=threadIdx.z; k<=threadIdx.z+2*kerneldim.z; k++)
    {
      for(uint j=threadIdx.y; j<=threadIdx.y+2*kerneldim.y; j++)
      {
        for(uint i=threadIdx.x; i<=threadIdx.x+2*kerneldim.x; i++)
        {
          _sum += sharedmem[k*sharedmemdim.x*sharedmemdim.y + j*sharedmemdim.x + i];    
        }
      }
    }   

    // store the result
    _out[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = (uchar) (_sum/neighborsize);
  }
}


// Kernel
__global__ void kernel_MeanFilter_F2F(float *_in, float *_out, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize)
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
  printf("tid : %d", tid); 
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
    // sum neighbor pixel values
    float _sum = 0;
    for(uint k=threadIdx.z; k<=threadIdx.z+2*kerneldim.z; k++)
    {
      for(uint j=threadIdx.y; j<=threadIdx.y+2*kerneldim.y; j++)
      {
        for(uint i=threadIdx.x; i<=threadIdx.x+2*kerneldim.x; i++)
        {
          _sum += sharedmem[k*sharedmemdim.x*sharedmemdim.y + j*sharedmemdim.x + i];    
        }
      }
    }   

    // store the result
    _out[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = _sum/neighborsize;
  }

/*

  //
  // Ver 1.0 : without using shared memory
  //  
  uint bidx = blockIdx.x;
  uint bidy = blockIdx.y%griddim.y;
  uint bidz = (uint)((blockIdx.y)/griddim.y);

  uint beidx = bidx*blockDim.x;
  uint beidy = bidy*blockDim.y;
  uint beidz = bidz*blockDim.z;

  // global index of the current voxel in the input volume
  uint idx = beidx + threadIdx.x;
  uint idy = beidy + threadIdx.y;
  uint idz = beidz + threadIdx.z;
 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {

    // sum neighbor pixel values
    int i, j, k;
    uint cx, cy, cz, iddx; 
    float nNb = (2*kerneldim.x+1)*(2*kerneldim.y+1)*(2*kerneldim.z+1); 
    float _sum = 0;
   
    for(k=(int)idz-(int)kerneldim.z; k<=(int)idz+(int)kerneldim.z; k++)
    {   
      for(j=(int)idy-(int)kerneldim.y; j<=(int)idy+(int)kerneldim.y; j++)
      {       
        for(i=(int)idx-(int)kerneldim.x; i<=(int)idx+(int)kerneldim.x; i++)
        {         
          // Neumann boundary condition
          cx = (uint) min(max(0,i),imagedim.x-1);
          cy = (uint) min(max(0,j),imagedim.y-1);
          cz = (uint) min(max(0,k),imagedim.z-1);

          iddx = cz*imagedim.x*imagedim.y + cy*imagedim.x + cx;
          _sum += _in[iddx];      
        }  
      }
    }   
     
    // store the result
    float val = _sum/nNb;
    _out[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = val;
  }

*/

}

__global__ void kernel_MeanFilter_U2U(uchar *_in, uchar *_out, dim3 imagedim, dim3 griddim, dim3 kerneldim, dim3 sharedmemdim, uint nUpdatePerThread, float neighborsize)
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
     
      sharedmem[sid] = (float)_in[gid];
    }
  }

  __syncthreads();


  // global index of the current voxel in the input volume
  uint idx = beidx + threadIdx.x;
  uint idy = beidy + threadIdx.y;
  uint idz = beidz + threadIdx.z;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    float _sum = 0;
    for(uint k=threadIdx.z; k<=threadIdx.z+2*kerneldim.z; k++)
    {
      for(uint j=threadIdx.y; j<=threadIdx.y+2*kerneldim.y; j++)
      {
        for(uint i=threadIdx.x; i<=threadIdx.x+2*kerneldim.x; i++)
        {
          _sum += sharedmem[k*sharedmemdim.x*sharedmemdim.y + j*sharedmemdim.x + i];    
        }
      }
    }   

    // store the result
    _sum /= neighborsize;
    _out[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = _sum;
  }


/*
  //
  // Ver 1.0 : without using shared memory
  //  
  uint bidx = blockIdx.x;
  uint bidy = blockIdx.y%griddim.y;
  uint bidz = (uint)((blockIdx.y)/griddim.y);

  uint beidx = bidx*blockDim.x;
  uint beidy = bidy*blockDim.y;
  uint beidz = bidz*blockDim.z;

  // global index of the current voxel in the input volume
  uint idx = beidx + threadIdx.x;
  uint idy = beidy + threadIdx.y;
  uint idz = beidz + threadIdx.z;
 
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    // sum neighbor pixel values
    int i, j, k;
    uint cx, cy, cz, iddx; 
    float nNb = (2*kerneldim.x+1)*(2*kerneldim.y+1)*(2*kerneldim.z+1); 
    float _sum = 0;
   
    for(k=(int)idz-(int)kerneldim.z; k<=(int)idz+(int)kerneldim.z; k++)
    {   
      for(j=(int)idy-(int)kerneldim.y; j<=(int)idy+(int)kerneldim.y; j++)
      {       
        for(i=(int)idx-(int)kerneldim.x; i<=(int)idx+(int)kerneldim.x; i++)
        {         
          // Neumann boundary condition
          cx = (uint) min(max(0,i),imagedim.x-1);
          cy = (uint) min(max(0,j),imagedim.y-1);
          cz = (uint) min(max(0,k),imagedim.z-1);

          iddx = cz*imagedim.x*imagedim.y + cy*imagedim.x + cx;
          _sum += _in[iddx];      
        }  
      }
    }   
     
    // store the result
    float val = _sum/nNb;
    _out[idz*imagedim.x*imagedim.y + idy*imagedim.x + idx] = val;
  }
*/
}
