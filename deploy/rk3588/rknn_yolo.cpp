// ============================================================================
//  rknn_yolo.cpp —— YOLOv5-5.0 受电弓检测, RK3588 NPU C++ 部署示例
//
//  ⚠️ RK3588 用的是 Rockchip 自研 NPU, 【不是】TensorRT。
//     .engine 文件在这里完全没用, 必须走 ONNX -> RKNN 转换 (见 convert_rknn.py)。
//
//  依赖:
//    头文件  rknn_api.h      <- rknpu2/runtime/RK3588/Linux/librknn_api/include
//    动态库  librknnrt.so    <- rknpu2/runtime/RK3588/Linux/librknn_api/aarch64
//    (板子上一般已经装好: /usr/lib/librknnrt.so)
//
//  编译: 见同目录 CMakeLists.txt
//  运行: ./rknn_yolo best_rk3588.rknn test.jpg out.jpg
//
//  与本项目 TensorRT 版本的等价性:
//    前处理 letterbox / 解码 / NMS 三处逻辑与 yolo_trt.cpp、infer_trt.py 完全一致,
//    后者已用 verify_cpp_logic.py 做过数值验证 (坐标偏差 6.1e-05 px)。
// ============================================================================
#include "rknn_api.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------------------------------- 常量区
// 必须与训练/导出严格一致
static const int   NET_H     = 640;             // 网络输入高 (历史遗留: 640 是高)
static const int   NET_W     = 480;             // 网络输入宽
static const int   NUM_CLS   = 2;
static const int   NUM_ANCH  = 3;
static const int   NUM_NO    = 5 + NUM_CLS;     // 7
static const float PAD_VALUE = 114.f;
static const float CONF_THRES = 0.25f;
static const float IOU_THRES  = 0.45f;

static const char* CLASS_NAMES[NUM_CLS] = { "pantograph", "contact point" };

static const float ANCHORS[3][NUM_ANCH][2] = {
    { { 10.f,  13.f}, { 16.f,  30.f}, { 33.f,  23.f} },   // P3/8
    { { 30.f,  61.f}, { 62.f,  45.f}, { 59.f, 119.f} },   // P4/16
    { {116.f,  90.f}, {156.f, 198.f}, {373.f, 326.f} },   // P5/32
};
static const int STRIDES[3]  = { 8, 16, 32 };
static const int GRID_NY[3]  = { 80, 40, 20 };
static const int GRID_NX[3]  = { 60, 30, 15 };

// ----------------------------------------------------------------- 工具
#define RKNN_CHK(call, what)                                                  \
    do {                                                                      \
        int _r = (call);                                                      \
        if (_r != RKNN_SUCC) {                                                \
            std::cerr << what << " 失败, ret=" << _r << std::endl;            \
            return false;                                                     \
        }                                                                     \
    } while (0)

static inline float sigmoid_f(float x) { return 1.f / (1.f + std::exp(-x)); }

static inline float iou_xyxy(const float* a, const float* b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float iw = std::max(0.f, x2 - x1);
    const float ih = std::max(0.f, y2 - y1);
    const float inter = iw * ih;
    const float ar = (a[2] - a[0]) * (a[3] - a[1]);
    const float br = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (ar + br - inter + 1e-7f);
}

struct Detection { float x1, y1, x2, y2, score; int cls; };
struct LetterboxInfo { float ratio, dw, dh; };

// ----------------------------------------------------------------- 前处理
// 与 yolo_trt.cpp 完全一致 (逐像素对齐 v5.0 官方 letterbox)
static cv::Mat letterbox(const cv::Mat& src, LetterboxInfo& info) {
    const int h = src.rows, w = src.cols;
    const float r = std::min(static_cast<float>(NET_H) / h,
                             static_cast<float>(NET_W) / w);
    const int newW = static_cast<int>(std::round(w * r));
    const int newH = static_cast<int>(std::round(h * r));
    const float dw = (NET_W - newW) / 2.f;
    const float dh = (NET_H - newH) / 2.f;

    cv::Mat resized = src;
    if (w != newW || h != newH)
        cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    const int top    = static_cast<int>(std::round(dh - 0.1f));
    const int left   = static_cast<int>(std::round(dw - 0.1f));

    cv::Mat canvas(NET_H, NET_W, CV_8UC3,
                   cv::Scalar(PAD_VALUE, PAD_VALUE, PAD_VALUE));
    resized.copyTo(canvas(cv::Rect(left, top, resized.cols, resized.rows)));

    info.ratio = r; info.dw = dw; info.dh = dh;
    return canvas;
}

// ----------------------------------------------------------------- 输出头几何
// RKNN 的输出轴顺序【不保证】与 ONNX 相同 (fmt / 内部 permute 都会改),
// 所以这里不写死下标, 而是按「语义」把每个轴认出来。
struct HeadGeom {
    int lvl = -1;                 // 0=P3/8, 1=P4/16, 2=P5/32
    int ndim = 0;                 // 5 或 3
    int na = 0, ny = 0, nx = 0, no = 0;
    int axNa = 0, axNy = 0, axNx = 0, axNo = 0;   // 5D 时: 各语义轴在 dims 里的下标
    int flatNa = 0;               // 3D 时: 会含 na 的那个展平轴下标
    size_t stride[8] = {0};       // 各轴元素步长 (第 0 轴最大)
};

// 按「无序对」判定属于哪一层
static int levelOfPair(int a, int b) {
    for (int l = 0; l < 3; ++l) {
        const int ny = GRID_NY[l], nx = GRID_NX[l];
        if ((a == ny && b == nx) || (a == nx && b == ny)) return l;
    }
    return -1;
}

static bool buildGeom(const rknn_tensor_attr& attr, HeadGeom& g) {
    const int nd = static_cast<int>(attr.n_dims);
    std::vector<int> d(nd);
    for (int i = 0; i < nd; ++i) d[i] = static_cast<int>(attr.dims[i]);

    std::cout << "    out: n_dims=" << nd << " dims=[";
    for (int i = 0; i < nd; ++i) std::cout << d[i] << (i + 1 < nd ? "," : "");
    std::cout << "] fmt=" << attr.fmt << " type=" << attr.type
              << " qnt=" << attr.qnt_type << " scale=" << attr.scale
              << " zp=" << attr.zp << std::endl;

    if (nd == 5) {
        g.ndim = 5;
        int axNo = -1, axNa = -1;
        for (int i = 0; i < 5; ++i) {
            if (d[i] == NUM_NO) axNo = i;
            if (d[i] == NUM_ANCH && i != axNo) axNa = i;
        }
        if (axNo < 0 || axNa < 0) {
            std::cerr << "      无法在 5D 形状里定位 no=" << NUM_NO
                      << " / na=" << NUM_ANCH << " 轴" << std::endl;
            return false;
        }
        int r0 = -1, r1 = -1;
        for (int i = 0; i < 5; ++i) {
            if (i == axNo || i == axNa) continue;
            if (r0 < 0) r0 = i;
            else        r1 = i;
        }
        const int lvl = levelOfPair(d[r0], d[r1]);
        if (lvl < 0) {
            std::cerr << "      两个空间轴 (" << d[r0] << "," << d[r1]
                      << ") 不匹配任何预期 grid" << std::endl;
            return false;
        }
        g.lvl = lvl;
        g.ny = GRID_NY[lvl];
        g.nx = GRID_NX[lvl];
        g.na = NUM_ANCH;
        g.no = NUM_NO;
        g.axNo = axNo;
        g.axNa = axNa;
        // ny > nx 恒成立 (80>60 / 40>30 / 20>15)
        if (d[r0] > d[r1]) { g.axNy = r0; g.axNx = r1; }
        else               { g.axNy = r1; g.axNx = r0; }

        size_t st = 1;
        for (int i = 4; i >= 0; --i) { g.stride[i] = st; st *= d[i]; }
        return true;
    }

    if (nd == 3) {
        // 展平版: [1, N, no] 的某种排列; N = na*ny*nx
        g.ndim = 3;
        int axNo = -1;
        for (int i = 0; i < 3; ++i) if (d[i] == NUM_NO) axNo = i;
        if (axNo < 0) {
            std::cerr << "      3D 形状里找不到 no=" << NUM_NO << " 轴" << std::endl;
            return false;
        }
        int axN = -1;
        for (int i = 0; i < 3; ++i)
            if (i != axNo && d[i] != 1) axN = i;
        if (axN < 0) {
            std::cerr << "      3D 形状里找不到展平轴" << std::endl;
            return false;
        }
        int lvl = -1;
        for (int l = 0; l < 3; ++l)
            if (d[axN] == NUM_ANCH * GRID_NY[l] * GRID_NX[l]) lvl = l;
        if (lvl < 0) {
            std::cerr << "      展平轴长度 " << d[axN]
                      << " 不匹配 na*ny*nx (应为 14400/3600/900)" << std::endl;
            return false;
        }
        g.lvl = lvl;
        g.ny = GRID_NY[lvl];
        g.nx = GRID_NX[lvl];
        g.na = NUM_ANCH;
        g.no = NUM_NO;
        g.axNo = axNo;
        g.flatNa = axN;

        size_t st = 1;
        for (int i = 2; i >= 0; --i) { g.stride[i] = st; st *= d[i]; }
        return true;
    }

    std::cerr << "      不支持的输出维度 n_dims=" << nd
              << " (本示例支持 5D 与 3D)" << std::endl;
    return false;
}

// 取 head 里 (a, y, x, c) 的值 —— 与 reshape 顺序无关
static inline float headAt(const HeadGeom& g, const float* p, int a, int y, int x, int c) {
    if (g.ndim == 5) {
        size_t off = static_cast<size_t>(a) * g.stride[g.axNa]
                   + static_cast<size_t>(y) * g.stride[g.axNy]
                   + static_cast<size_t>(x) * g.stride[g.axNx]
                   + static_cast<size_t>(c) * g.stride[g.axNo];
        return p[off];
    }
    // 3D: 展平轴内部按 (a, y, x) 行主序
    const size_t inner = (static_cast<size_t>(a) * g.ny + y) * g.nx + x;
    return p[inner * g.stride[g.flatNa] + static_cast<size_t>(c) * g.stride[g.axNo]];
}

// ----------------------------------------------------------------- 引擎
class RknnEngine {
public:
    ~RknnEngine() { if (ctx_) rknn_destroy(ctx_); }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "打不开模型: " << path << std::endl; return false; }
        model_.assign(std::istreambuf_iterator<char>(f),
                      std::istreambuf_iterator<char>());
        std::cout << "模型    : " << path << "  ("
                  << model_.size() / 1024.0 / 1024.0 << " MB)" << std::endl;

        // 可以传 uint64_t flags 指定 core mask, 例如:
        //   RKNN_FLAG_COLLECT_PERF_MASK | (1<<0)  <- 指定 NPU core 0
        RKNN_CHK(rknn_init(&ctx_, model_.data(), model_.size(), 0, nullptr),
                 "rknn_init");

        rknn_input_output_num io;
        RKNN_CHK(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)),
                 "RKNN_QUERY_IN_OUT_NUM");
        nIn_  = io.n_input;
        nOut_ = io.n_output;
        std::cout << "IO      : " << nIn_ << " 输入, " << nOut_ << " 输出" << std::endl;

        inAttr_.resize(nIn_);
        for (uint32_t i = 0; i < nIn_; ++i) {
            std::memset(&inAttr_[i], 0, sizeof(rknn_tensor_attr));
            inAttr_[i].index = i;
            RKNN_CHK(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inAttr_[i],
                                sizeof(rknn_tensor_attr)), "QUERY_INPUT_ATTR");
            std::cout << "    in : n_dims=" << inAttr_[i].n_dims << " dims=[";
            for (uint32_t k = 0; k < inAttr_[i].n_dims; ++k)
                std::cout << inAttr_[i].dims[k]
                          << (k + 1 < inAttr_[i].n_dims ? "," : "");
            std::cout << "] fmt=" << inAttr_[i].fmt
                      << " type=" << inAttr_[i].type << std::endl;
        }
        if (nIn_ != 1) { std::cerr << "本示例只处理单输入模型" << std::endl; return false; }

        outAttr_.resize(nOut_);
        geoms_.resize(nOut_);
        for (uint32_t i = 0; i < nOut_; ++i) {
            std::memset(&outAttr_[i], 0, sizeof(rknn_tensor_attr));
            outAttr_[i].index = i;
            RKNN_CHK(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &outAttr_[i],
                                sizeof(rknn_tensor_attr)), "QUERY_OUTPUT_ATTR");
            if (!buildGeom(outAttr_[i], geoms_[i])) return false;
        }

        // 检查是否三层都齐了
        bool seen[3] = { false, false, false };
        for (size_t i = 0; i < geoms_.size(); ++i)
            if (geoms_[i].lvl >= 0) seen[geoms_[i].lvl] = true;
        if (!(seen[0] && seen[1] && seen[2])) {
            std::cerr << "输出头不完整 (需要 P3/8 + P4/16 + P5/32 三个), "
                         "请检查 RKNN 是否合并/删除了输出" << std::endl;
            return false;
        }
        return true;
    }

    // 输入 img 为已 letterbox 的 BGR 图 (NET_H x NET_W)
    bool infer(const cv::Mat& lbImg, std::vector<Detection>& dets, float r,
               float dw, float dh, int w0, int h0)
    {
        // ---- 按 RKNN 声明的输入格式填 buffer ----
        // pass_through=0 + FLOAT32: 由 RKNN 按 config 的 mean/std 归一化
        std::vector<float> blob(static_cast<size_t>(NET_H) * NET_W * 3);
        const bool nhwc = (inAttr_[0].fmt == RKNN_TENSOR_NHWC);
        const size_t plane = static_cast<size_t>(NET_H) * NET_W;

        for (int y = 0; y < NET_H; ++y) {
            const cv::Vec3b* row = lbImg.ptr<cv::Vec3b>(y);
            for (int x = 0; x < NET_W; ++x) {
                const cv::Vec3b& p = row[x];
                // 训练时 BGR->RGB, 这里必须给 RGB
                const float v0 = static_cast<float>(p[2]);   // R
                const float v1 = static_cast<float>(p[1]);   // G
                const float v2 = static_cast<float>(p[0]);   // B
                if (nhwc) {
                    const size_t o = (static_cast<size_t>(y) * NET_W + x) * 3;
                    blob[o + 0] = v0;
                    blob[o + 1] = v1;
                    blob[o + 2] = v2;
                } else {
                    const size_t o = static_cast<size_t>(y) * NET_W + x;
                    blob[0 * plane + o] = v0;
                    blob[1 * plane + o] = v1;
                    blob[2 * plane + o] = v2;
                }
            }
        }

        rknn_input inputs[1];
        std::memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].fmt  = inAttr_[0].fmt;
        inputs[0].size = blob.size() * sizeof(float);
        inputs[0].buf  = blob.data();
        inputs[0].pass_through = 0;
        RKNN_CHK(rknn_inputs_set(ctx_, 1, inputs), "rknn_inputs_set");

        RKNN_CHK(rknn_run(ctx_, nullptr), "rknn_run");

        std::vector<rknn_output> outs(nOut_);
        std::memset(outs.data(), 0, sizeof(rknn_output) * nOut_);
        for (uint32_t i = 0; i < nOut_; ++i) outs[i].want_float = 1;   // 自动反量化
        RKNN_CHK(rknn_outputs_get(ctx_, nOut_, outs.data(), nullptr),
                 "rknn_outputs_get");

        // ---- 解码 (与 TensorRT 版逻辑完全相同) ----
        std::vector<Detection> cand;
        for (uint32_t i = 0; i < nOut_; ++i) {
            const HeadGeom& g = geoms_[i];
            if (g.lvl < 0) continue;
            const float* p = static_cast<const float*>(outs[i].buf);
            const int stride = STRIDES[g.lvl];

            for (int a = 0; a < g.na; ++a)
                for (int y = 0; y < g.ny; ++y)
                    for (int x = 0; x < g.nx; ++x) {
                        const float obj = sigmoid_f(headAt(g, p, a, y, x, 4));
                        if (obj <= CONF_THRES) continue;

                        int   bestC = 0;
                        float bestP = headAt(g, p, a, y, x, 5);
                        for (int c = 1; c < NUM_CLS; ++c) {
                            const float v = headAt(g, p, a, y, x, 5 + c);
                            if (v > bestP) { bestP = v; bestC = c; }
                        }
                        const float score = obj * sigmoid_f(bestP);
                        if (score <= CONF_THRES) continue;

                        const float cx = (sigmoid_f(headAt(g, p, a, y, x, 0)) * 2.f
                                          - 0.5f + x) * stride;
                        const float cy = (sigmoid_f(headAt(g, p, a, y, x, 1)) * 2.f
                                          - 0.5f + y) * stride;
                        const float bw = std::pow(sigmoid_f(headAt(g, p, a, y, x, 2))
                                                  * 2.f, 2.f) * ANCHORS[g.lvl][a][0];
                        const float bh = std::pow(sigmoid_f(headAt(g, p, a, y, x, 3))
                                                  * 2.f, 2.f) * ANCHORS[g.lvl][a][1];

                        Detection d;
                        d.x1 = cx - bw / 2.f;
                        d.y1 = cy - bh / 2.f;
                        d.x2 = cx + bw / 2.f;
                        d.y2 = cy + bh / 2.f;
                        d.score = score;
                        d.cls = bestC;
                        cand.push_back(d);
                    }
        }

        RKNN_CHK(rknn_outputs_release(ctx_, nOut_, outs.data()),
                 "rknn_outputs_release");

        // ---- NMS ----
        std::sort(cand.begin(), cand.end(),
                  [](const Detection& a, const Detection& b) { return a.score > b.score; });
        std::vector<char> dead(cand.size(), 0);
        dets.clear();
        for (size_t i = 0; i < cand.size(); ++i) {
            if (dead[i]) continue;
            dets.push_back(cand[i]);
            const float A[4] = { cand[i].x1, cand[i].y1, cand[i].x2, cand[i].y2 };
            for (size_t j = i + 1; j < cand.size(); ++j) {
                if (dead[j] || cand[j].cls != cand[i].cls) continue;
                const float B[4] = { cand[j].x1, cand[j].y1, cand[j].x2, cand[j].y2 };
                if (iou_xyxy(A, B) > IOU_THRES) dead[j] = 1;
            }
        }

        // ---- 回到原图坐标 ----
        for (size_t i = 0; i < dets.size(); ++i) {
            Detection& d = dets[i];
            d.x1 = std::max(0.f, std::min((d.x1 - dw) / r, static_cast<float>(w0)));
            d.x2 = std::max(0.f, std::min((d.x2 - dw) / r, static_cast<float>(w0)));
            d.y1 = std::max(0.f, std::min((d.y1 - dh) / r, static_cast<float>(h0)));
            d.y2 = std::max(0.f, std::min((d.y2 - dh) / r, static_cast<float>(h0)));
        }
        return true;
    }

private:
    rknn_context            ctx_   = 0;
    std::vector<char>       model_;
    uint32_t                nIn_ = 0, nOut_ = 0;
    std::vector<rknn_tensor_attr> inAttr_, outAttr_;
    std::vector<HeadGeom>   geoms_;
};

// ----------------------------------------------------------------- 画框
static void draw(cv::Mat& img, const std::vector<Detection>& dets) {
    const cv::Scalar colors[2] = { cv::Scalar(60, 76, 231), cv::Scalar(74, 195, 139) };
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        const cv::Scalar c = colors[d.cls % 2];
        cv::rectangle(img, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                           cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), c, 2);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s %.2f", CLASS_NAMES[d.cls], d.score);
        int base = 0;
        const cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::rectangle(img,
                      cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1) - ts.height - 6),
                      cv::Point(static_cast<int>(d.x1) + ts.width + 4, static_cast<int>(d.y1)),
                      c, -1);
        cv::putText(img, buf,
                    cv::Point(static_cast<int>(d.x1) + 2, static_cast<int>(d.y1) - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ----------------------------------------------------------------- 主流程
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0]
                  << " <model.rknn> <图片或目录> [输出目录]\n"
                     "例:   " << argv[0]
                  << " best_rk3588.rknn ./val ./out\n";
        return 1;
    }
    const std::string modelPath = argv[1];
    const std::string source    = argv[2];
    const std::string outDir    = (argc > 3) ? argv[3] : "";

    RknnEngine engine;
    if (!engine.load(modelPath)) return 2;

    std::vector<cv::String> files;
    if (source.find(".jpg") != std::string::npos ||
        source.find(".png") != std::string::npos) {
        files.push_back(source);
    } else {
        cv::glob(source + "/*.jpg", files, false);
    }
    if (files.empty()) { std::cerr << "没找到图片: " << source << std::endl; return 3; }
    std::cout << "图片    : " << files.size() << " 张\n" << std::endl;

    int total = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat img = cv::imread(files[i]);
        if (img.empty()) { std::cerr << "读图失败: " << files[i] << std::endl; continue; }

        LetterboxInfo li;
        cv::Mat lb = letterbox(img, li);

        std::vector<Detection> dets;
        if (!engine.infer(lb, dets, li.ratio, li.dw, li.dh, img.cols, img.rows))
            return 4;

        std::cout << "[" << (i + 1) << "/" << files.size() << "] " << files[i]
                  << "  检出 " << dets.size() << " 个" << std::endl;
        for (size_t k = 0; k < dets.size(); ++k) {
            const Detection& d = dets[k];
            std::cout << "      " << CLASS_NAMES[d.cls]
                      << "  conf=" << d.score
                      << "  box=(" << static_cast<int>(d.x1) << "," << static_cast<int>(d.y1)
                      << ")-(" << static_cast<int>(d.x2) << "," << static_cast<int>(d.y2)
                      << ")" << std::endl;
        }
        total += static_cast<int>(dets.size());

        if (!outDir.empty()) {
            draw(img, dets);
            size_t slash = files[i].find_last_of("/\\");
            std::string baseName = (slash == std::string::npos) ? files[i]
                                                                : files[i].substr(slash + 1);
            cv::imwrite(outDir + "/" + baseName, img);
        }
    }

    std::cout << "\n合计检出 " << total << " 个目标" << std::endl;
    if (!outDir.empty()) std::cout << "标注图已存: " << outDir << std::endl;
    return 0;
}
