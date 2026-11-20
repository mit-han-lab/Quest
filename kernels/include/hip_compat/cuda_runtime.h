#pragma once
#include <hip/hip_runtime.h>

// The kernels call the CUDA runtime by name. Every symbol below is a one-to-one rename in
// HIP, so aliasing here keeps the kernel sources identical between the two backends.
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaFuncAttributes = hipFuncAttributes;
using cudaMemcpyKind = hipMemcpyKind;

#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor \
	hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaLaunchKernel hipLaunchKernel
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
// hipFuncSetAttribute takes const void*, where cudaFuncSetAttribute is a template that
// accepts a typed kernel pointer; the cast keeps the call sites unchanged.
#define cudaFuncSetAttribute(func, attr, value) \
	hipFuncSetAttribute((const void*)(func), attr, value)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
	hipOccupancyMaxActiveBlocksPerMultiprocessor
