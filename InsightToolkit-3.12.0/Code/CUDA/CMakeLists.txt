#
# CUDA for itk
#

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT ON)
else()
  set(CUDA_64_BIT_DEVICE_CODE_DEFAULT OFF)
endif()
OPTION(CUDA_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${CUDA_64_BIT_DEVICE_CODE_DEFAULT})


FILE(GLOB_RECURSE itkCudaSRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *.cu)
FILE(GLOB_RECURSE itkCudaHDR RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *.h)
FILE(GLOB_RECURSE itkCxxSRC  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *.cxx)

SOURCE_GROUP("CUDA Files" FILES ${itkCudaSRC})

#MESSAGE("Source : " ${itkCudaSRC})

SET(targetSRC "")
SET(buildFlag "")

# choose DEBUG/RELEASE/EMUDEBUG/EMURELEASE
IF(NOT CUDA_BUILD_TYPE)
	SET(CUDA_BUILD_TYPE "RELEASE" CACHE STRING "CUDA build type: DEBUG/RELEASE/EMUDEBUG/EMURELEASE")
ENDIF(NOT CUDA_BUILD_TYPE)
	
# Search for the CUDA SDK
IF (NOT CUDA_SDK_DIR)
	SET(CUDA_SDK_DIR "C:/Program Files/NVIDIA Corporation/NVIDIA CUDA SDK/"
		CACHE PATH "CUDA SDK directory")
  IF (NOT EXISTS ${CUDA_SDK_DIR})
    MESSAGE(FATAL_ERROR "Specify CUDA SDK directory")
  ENDIF (NOT EXISTS ${CUDA_SDK_DIR})
ENDIF (NOT CUDA_SDK_DIR)

#MESSAGE("CUDA BUILD TYPE : " ${CUDA_BUILD_TYPE})

# MATCHES will return true if left contains right texts (not exactly equal)
IF(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
	SET(buildFlag ${buildFlag} -cuda -g -D_DEBUG -deviceemu)
ELSEIF(CUDA_BUILD_TYPE MATCHES "EMURELEASE")
	SET(buildFlag ${buildFlag} -cuda -O -deviceemu)
ELSEIF(CUDA_BUILD_TYPE MATCHES "DEBUG")
	SET(buildFlag ${buildFlag} -cuda -g -D_DEBUG)
ELSEIF(CUDA_BUILD_TYPE MATCHES "RELEASE")
	SET(buildFlag ${buildFlag} -cuda -O)
ELSE(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
	SET(buildFlag ${buildFlag} -cuda -O)
ENDIF(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
	
# Find CUDA directories/files

# CUDA_NVCC
IF(UNIX)
IF(NOT CUDA_NVCC)
		FIND_PROGRAM(CUDA_NVCC 
			nvcc
			PATHS $ENV{CUDA_BIN_PATH}
		)
		IF(NOT CUDA_NVCC)
			MESSAGE(FATAL_ERROR "nvcc is not found.")
		ELSE(NOT CUDA_NVCC)
			MARK_AS_ADVANCED(CUDA_NVCC)
		ENDIF(NOT CUDA_NVCC)
	ENDIF(NOT CUDA_NVCC)
ELSE(UNIX)
	IF(NOT CUDA_NVCC)
		FIND_PROGRAM(CUDA_NVCC 
			nvcc.exe
			PATHS $ENV{CUDA_BIN_PATH}
		)
		IF(NOT CUDA_NVCC)
			MESSAGE(FATAL_ERROR "nvcc is not found.")
		ELSE(NOT CUDA_NVCC)
			MARK_AS_ADVANCED(CUDA_NVCC)
		ENDIF(NOT CUDA_NVCC)
	ENDIF(NOT CUDA_NVCC)
ENDIF(UNIX)

# CUDA INCLUDE PATH
IF(NOT CUDA_INCLUDE_DIR)
	FIND_PATH(CUDA_INCLUDE_DIR
		device_functions.h
		PATHS $ENV{CUDA_INC_PATH}
	)  
	IF(NOT CUDA_INCLUDE_DIR)
		MESSAGE(FATAL_ERROR "Could not find CUDA headers")
	ELSE(NOT CUDA_INCLUDE_DIR)
		MARK_AS_ADVANCED(CUDA_INCLUDE_DIR)		
	ENDIF(NOT CUDA_INCLUDE_DIR)
ENDIF(NOT CUDA_INCLUDE_DIR)


# CUDA LIB PATH
IF(UNIX)
	IF(NOT CUDA_LIB_DIR)
		FIND_PATH(CUDA_LIB_DIR
			libcudart.so
			PATHS $ENV{CUDA_LIB_PATH}
		)  
		IF(NOT CUDA_LIB_DIR)
			MESSAGE(FATAL_ERROR "Could not find CUDA libraries")
		ELSE(NOT CUDA_LIB_DIR)
			MARK_AS_ADVANCED(CUDA_LIB_DIR)
		ENDIF(NOT CUDA_LIB_DIR)
	ENDIF(NOT CUDA_LIB_DIR)	
ELSE(UNIX)
	IF(NOT CUDA_LIB_DIR)
		FIND_PATH(CUDA_LIB_DIR
			cudart.lib
			PATHS $ENV{CUDA_LIB_PATH}
		)  
		IF(NOT CUDA_LIB_DIR)
			MESSAGE(FATAL_ERROR "Could not find CUDA libraries")
		ELSE(NOT CUDA_LIB_DIR)
			MARK_AS_ADVANCED(CUDA_LIB_DIR)
		ENDIF(NOT CUDA_LIB_DIR)
	ENDIF(NOT CUDA_LIB_DIR)	
ENDIF(UNIX)
# Collect CUDA lib for link

IF(UNIX)
	IF(NOT CUDA_UNIX_64_BIT)
	    SET(CUDA_UNIX_64_BIT "ON")
	    
	    MARK_AS_ADVANCED(CUDA_UNIX_64_BIT)
	ENDIF(NOT CUDA_UNIX_64_BIT)

	IF(CUDA_UNIX_64_BIT)
	    IF(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/*emu.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_SDK_DIR}/common/lib/linux/*64_emu.a ${CUDA_SDK_DIR}/common/lib/linux/*64D_emu.a ${CUDA_SDK_DIR}/lib/*.a)
	    ELSEIF(CUDA_BUILD_TYPE MATCHES "EMURELEASE")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/*emu.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_SDK_DIR}/common/lib/linux/*64_emu.a ${CUDA_SDK_DIR}/common/lib/linux/*64D_emu.a ${CUDA_SDK_DIR}/lib/*.a)
	    ELSEIF(CUDA_BUILD_TYPE MATCHES "DEBUG")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/libcublas.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_LIB_DIR}/libcufft.so ${CUDA_SDK_DIR}/common/lib/linux/libcudpp64.a ${CUDA_SDK_DIR}/common/lib/linux/libcudpp64D.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW_x86_64.a ${CUDA_SDK_DIR}/lib/*.a)
	    ELSEIF(CUDA_BUILD_TYPE MATCHES "RELEASE")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/libcublas.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_LIB_DIR}/libcufft.so ${CUDA_SDK_DIR}/common/lib/linux/libcudpp64.a ${CUDA_SDK_DIR}/common/lib/linux/libcudpp64D.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW_x86_64.a ${CUDA_SDK_DIR}/lib/*.a)
	    ELSE(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/libcublas.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_LIB_DIR}/libcufft.so ${CUDA_SDK_DIR}/common/lib/linux/libcudpp64.a ${CUDA_SDK_DIR}/common/lib/linux/libcudpp64D.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW_x86_64.a ${CUDA_SDK_DIR}/lib/*.a)
	    ENDIF(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
	ELSE(CUDA_UNIX_64_BIT)
	    IF(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/*emu.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_SDK_DIR}/common/lib/linux/libcudppD_emu.a ${CUDA_SDK_DIR}/common/lib/linux/libcudpp_emu.a ${CUDA_SDK_DIR}/lib/*.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW.a)
	    ELSEIF(CUDA_BUILD_TYPE MATCHES "EMURELEASE")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/*emu.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_SDK_DIR}/common/lib/linux/libcudppD_emu.a ${CUDA_SDK_DIR}/common/lib/linux/libcudpp_emu.a ${CUDA_SDK_DIR}/lib/*.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW.a)
	    ELSEIF(CUDA_BUILD_TYPE MATCHES "DEBUG")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/libcublas.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_LIB_DIR}/libcufft.so ${CUDA_SDK_DIR}/common/lib/linux/libcudpp.a  ${CUDA_SDK_DIR}/common/lib/linux/libcudppD.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW.a ${CUDA_SDK_DIR}/lib/*.a)
	    ELSEIF(CUDA_BUILD_TYPE MATCHES "RELEASE")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/libcublas.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_LIB_DIR}/libcufft.so ${CUDA_SDK_DIR}/common/lib/linux/libcudpp.a  ${CUDA_SDK_DIR}/common/lib/linux/libcudppD.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW.a ${CUDA_SDK_DIR}/lib/*.a)
	    ELSE(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
		    FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/libcublas.so ${CUDA_LIB_DIR}/libcudart.so ${CUDA_LIB_DIR}/libcufft.so ${CUDA_SDK_DIR}/common/lib/linux/libcudpp.a  ${CUDA_SDK_DIR}/common/lib/linux/libcudppD.a ${CUDA_SDK_DIR}/common/lib/linux/libGLEW.a ${CUDA_SDK_DIR}/lib/*.a)
	    ENDIF(CUDA_BUILD_TYPE MATCHES "EMUDEBUG")
	ENDIF(CUDA_UNIX_64_BIT)

ELSE(UNIX)	
	FILE(GLOB_RECURSE CUDA_LIB ${CUDA_LIB_DIR}/*.lib ${CUDA_SDK_DIR}/common/lib/*.lib)
ENDIF(UNIX)

MARK_AS_ADVANCED(CUDA_LIB)

#MESSAGE("CudaLib: " ${CUDA_LIB})
	
# Update build flag
SET(buildFlag ${buildFlag} -I${CUDA_INCLUDE_DIR} -I${CUDA_SDK_DIR}/common/inc)
IF(WIN32)
	SET(buildFlag ${buildFlag} -DWIN32)
ENDIF(WIN32)

IF(CUDA_64_BIT_DEVICE_CODE)
    SET(buildFlag ${buildFlag} -m64)
ELSE()
    SET(buildFlag ${buildFlag} -m32)
ENDIF()

# Check if everything is ok	
#MESSAGE("Build Flag : " ${buildFlag})
#MESSAGE("Input file : " ${itkCudaSRC})
#MESSAGE("CUDA Lib dir : " ${CUDA_LIB_DIR})
#MESSAGE("CUDA Lib : " ${CUDA_LIB})
#MESSAGE("CUDA NVCC : " ${CUDA_NVCC})
#MESSAGE("ccbin_flags : " ${ccbin_flags})

# Custom build for CUDA files	
IF(UNIX)
	FOREACH(file ${itkCudaSRC})
		SET(gen_file "${CMAKE_CURRENT_BINARY_DIR}/${file}_gen.cpp")	
		ADD_CUSTOM_COMMAND(
			OUTPUT ${gen_file}
			COMMAND ${CUDA_NVCC} ARGS -cuda ${CMAKE_CURRENT_SOURCE_DIR}/${file} --compiler-bindir="/usr/bin/" ${buildFlag} -o ${gen_file} 
		)			
		SET(targetSRC ${targetSRC} ${gen_file})	
	ENDFOREACH(file)
ELSE(UNIX)
	FOREACH(file ${itkCudaSRC})
		SET(gen_file "${CMAKE_CURRENT_BINARY_DIR}/${file}_gen.cpp")	
		ADD_CUSTOM_COMMAND(
			OUTPUT ${gen_file}
			COMMAND ${CUDA_NVCC} ARGS -cuda ${CMAKE_CURRENT_SOURCE_DIR}/${file} --compiler-bindir="$(VCInstallDir)bin/" ${buildFlag} -o ${gen_file} 
		)			
		SET(targetSRC ${targetSRC} ${gen_file})	
	ENDFOREACH(file)
ENDIF(UNIX)

FOREACH(file ${itkCudaSRC})
	SET(gen_file "${CMAKE_CURRENT_BINARY_DIR}/${file}_gen.cubin")	
	ADD_CUSTOM_COMMAND(
		OUTPUT ${gen_file}
    	COMMAND ${CUDA_NVCC} ARGS -cubin -I${CUDA_INCLUDE_DIR} -I${CUDA_SDK_DIR}/common/inc ${CMAKE_CURRENT_SOURCE_DIR}/${file} 
	)	
ENDFOREACH(file)


# Include path
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR} ${CUDA_SDK_DIR}/common/inc)

#MESSAGE("target source : " ${targetSRC})

# Create library
ADD_LIBRARY(itkCuda ${targetSRC} ${itkCxxSRC} ${itkCudaHDR} ${itkCudaSRC})
TARGET_LINK_LIBRARIES(itkCuda ${CUDA_LIB})# ITKCommon ITKIO)
