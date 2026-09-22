// ============================================================================
//  cuda_runtime_api.h —— 【仅用于语法检查的桩头文件】
//  声明 deploy/*.cpp 里用到的那几个 CUDA Runtime 符号。
//  【不参与真正的编译, 不要拿去构建!】
// ============================================================================
#pragma once
#include <cstddef>

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1
} cudaError_t;

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct cudaGraph_st* cudaGraph_t;
typedef struct cudaGraphExec_st* cudaGraphExec_t;
typedef struct cudaGraphNode_st* cudaGraphNode_t;
typedef struct cudaUserObject_st* cudaUserObject_t;
typedef struct cudaArray_st* cudaArray_t;
typedef struct cudaMipmappedArray_st* cudaMipmappedArray_t;
typedef struct CUexternalMemory_st* cudaExternalMemory_t;
typedef struct CUexternalSemaphore_st* cudaExternalSemaphore_t;
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaHostFn_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

cudaError_t cudaMalloc(void** devPtr, std::size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
const char* cudaGetErrorString(cudaError_t error);
