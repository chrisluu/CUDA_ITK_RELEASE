PROJECT(ITK_CUDA_TEST)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# common

FILE(GLOB_RECURSE SRC *.cpp *.cxx *.c)
FILE(GLOB_RECURSE HDR *.h)


# depends on OS

IF (UNIX)

 MESSAGE(STATUS "NOW RUNNING LINUX MODE")
  
  # Find ITK.
  INCLUDE(FindITK)
  FIND_PACKAGE(ITK)
  IF(ITK_FOUND)
  	INCLUDE(${ITK_USE_FILE})
	MESSAGE("ITK_USE_FILE: " ${ITK_USE_FILE})
  ELSE(ITK_FOUND)
  	MESSAGE(FATAL_ERROR "Cannot build without ITK.  Please set ITK_DIR.")
  ENDIF(ITK_FOUND) 


  ADD_EXECUTABLE(test_meanmedianimagefilter test_MeanMedianImageFilter.cpp ${HDR}) 
  ADD_EXECUTABLE(test_discretegaussianimagefilter test_DiscreteGaussianImageFilter.cpp ${HDR}) 
  ADD_EXECUTABLE(test_hessianrecursivegaussianimagefilter test_HessianRecursiveGaussianImageFilter.cpp ${HDR})
  ADD_EXECUTABLE(test_derivativeimagefilter test_DerivativeImageFilter.cpp ${HDR}) 
 
  TARGET_LINK_LIBRARIES(test_meanmedianimagefilter ITKCommon ITKIO itkCuda -lcuda)
  TARGET_LINK_LIBRARIES(test_discretegaussianimagefilter ITKCommon ITKIO itkCuda -lcuda)
  TARGET_LINK_LIBRARIES(test_derivativeimagefilter ITKCommon ITKIO itkCuda -lcuda)
  TARGET_LINK_LIBRARIES(test_hessianrecursivegaussianimagefilter ITKCommon ITKIO itkCuda -lcuda)

ELSE (UNIX)

  MESSAGE(STATUS "NOW RUNNING WIN32 MODE")

  # for Visual C++ 2005
  SET( CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "/NODEFAULTLIBS:LIBCMT" )
  SET( CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "/NODEFAULTLIBS:MSVCRT" )	
  SET( CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} "/NODEFAULTLIBS:MSVCPRTD" )	

  ADD_EXECUTABLE(test_meanmedianimagefilter test_meanmedianimagefilter.cpp ${HDR}) 
  ADD_EXECUTABLE(test_discretegaussianimagefilter test_discretegaussianimagefilter.cpp ${HDR}) 
  ADD_EXECUTABLE(test_hessianrecursivegaussianimagefilter test_hessianrecursivegaussianimagefilter.cpp ${HDR})
  ADD_EXECUTABLE(test_derivativeimagefilter test_derivativeimagefilter.cpp ${HDR}) 
 
   #--------------------------------------------------------------------------------
  # Add ITK
  find_package(ITK REQUIRED)
  IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
  ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK.  Please set ITK_DIR.")
  ENDIF(ITK_FOUND) 
  
  include_directories(${ITK_INCLUDE_DIRS})
  
  TARGET_LINK_LIBRARIES(test_meanmedianimagefilter ITKCommon ITKIO itkCuda)
  TARGET_LINK_LIBRARIES(test_discretegaussianimagefilter ITKCommon ITKIO itkCuda)
  TARGET_LINK_LIBRARIES(test_derivativeimagefilter ITKCommon ITKIO itkCuda)
  TARGET_LINK_LIBRARIES(test_hessianrecursivegaussianimagefilter ITKCommon ITKIO itkCuda)

ENDIF (UNIX)
