// ============================================================================
//  rknn_api.h —— 【仅用于语法检查的桩头文件】
//
//  声明 deploy/rk3588/rknn_yolo.cpp 里用到的 RKNN API 符号, 用于在没有 rknpu2
//  的环境下做 -fsyntax-only 检查。
//
//  ⚠️ 真实头文件来自 rknpu2:
//       rknpu2/runtime/RK3588/Linux/librknn_api/include/rknn_api.h
//     本文件【不参与真正的编译, 不要拿去构建!】
//     其中的常量数值仅为占位, 与真实枚举值不一定一致。
// ============================================================================
#pragma once
#include <cstdint>

#define RKNN_SUCC 0
#define RKNN_FAIL -1
#define RKNN_MAX_DIMS 16
#define RKNN_MAX_NAME_LEN 256

typedef uint32_t rknn_context;

typedef enum {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16,
    RKNN_TENSOR_INT8,
    RKNN_TENSOR_UINT8,
    RKNN_TENSOR_INT32,
    RKNN_TENSOR_INT64
} rknn_tensor_type;

typedef enum {
    RKNN_TENSOR_NCHW = 0,
    RKNN_TENSOR_NHWC,
    RKNN_TENSOR_NC1HWC2,
    RKNN_TENSOR_UNDEFINED
} rknn_tensor_format;

typedef enum {
    RKNN_QUERY_IN_OUT_NUM = 0,
    RKNN_QUERY_INPUT_ATTR,
    RKNN_QUERY_OUTPUT_ATTR,
    RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR = 6,
    RKNN_QUERY_SDK_VERSION
} rknn_query_cmd;

typedef struct {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    char     name[RKNN_MAX_NAME_LEN];
    uint32_t n_elems;
    uint32_t size;
    rknn_tensor_format fmt;
    rknn_tensor_type   type;
    int32_t  qnt_type;
    int8_t   fl;
    int32_t  zp;
    float    scale;
} rknn_tensor_attr;

typedef struct {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

typedef struct {
    uint32_t index;
    void*    buf;
    uint32_t size;
    uint8_t  pass_through;
    rknn_tensor_type   type;
    rknn_tensor_format fmt;
} rknn_input;

typedef struct {
    void*    buf;
    uint32_t size;
    int      want_float;
    int      is_prealloc;
} rknn_output;

typedef struct {
    int fd;
    void* virt_addr;
    uint32_t size;
    uint32_t flags;
} rknn_tensor_mem;

int rknn_init(rknn_context* context, void* model, uint32_t size,
              uint32_t flag, void* extend);
int rknn_destroy(rknn_context context);
int rknn_query(rknn_context context, rknn_query_cmd cmd, void* info,
               uint32_t size);
int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[]);
int rknn_run(rknn_context context, void* extend);
int rknn_outputs_get(rknn_context context, uint32_t n_outputs,
                     rknn_output outputs[], void* extend);
int rknn_outputs_release(rknn_context context, uint32_t n_outputs,
                         rknn_output outputs[]);
rknn_tensor_mem* rknn_create_mem(rknn_context context, uint32_t size);
int rknn_set_io_mem(rknn_context context, rknn_tensor_mem* mem,
                    rknn_tensor_attr* attr);
int rknn_destroy_mem(rknn_context context, rknn_tensor_mem* mem);
