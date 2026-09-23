// ============================================================================
//  yolo_trt.cpp —— YOLOv5-5.0 , TensorRT C++ 
//
//  适用: Jetson 全系 (Nano/Xavier/Orin) 以及任意 x86 + NVIDIA GPU 的 Linux
//  模型: best.onnx -> best_fp16.engine / best_int8_minmax.engine
//  输入: 640 x 480   注意是「高=640, 宽=480」, 这是训练时的历史遗留口径,
//                    千万不要"顺手修正"成 480x640, 否则精度直接崩
//
//  编译: 见同目录 CMakeLists.txt        运行: ./yolo_trt <engine> <图片或目录>
//
//  ── 本文件与 Python 版 (infer_trt.py) 的等价性已用 verify_cpp_logic.py 验证 ──
//     坐标最大偏差 6.1e-05 px / 置信度 1.2e-07 / 框数与类别完全一致
//
//  ── TensorRT 10 API 变更提醒 (老教程全部gg) ──────────────────────────────
//     × context->enqueueV2(...)          -> 用 enqueueV3(stream)
//     × context->execute(1,...)          -> 用 enqueueV3 / executeV3
//     × engine->getNbBindings()          -> 用 getNbIOTensors()
//     × engine->getBindingIndex(name)    -> 用 getIOTensorName(i) 遍历
//     × runtime->deserializeCudaEngine(blob, size, pluginFactory)
//                                        -> 只剩 deserializeCudaEngine(blob, size)
// ============================================================================
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ----------------------------------------------------------------- 常量区
// 全部必须与训练/导出严格一致, 改一个都会导致检测结果错乱
static const int   NET_H     = 640;              // 网络输入高 (历史遗留)
static const int   NET_W     = 480;              // 网络输入宽
static const int   NUM_CLS   = 2;                // ['pantograph', 'contact point']
static const int   NUM_ANCH  = 3;                // 每个 grid 3 个 anchor
static const int   NUM_NO    = 5 + NUM_CLS;      // 4+1+nc = 7
static const float PAD_VALUE = 114.f;            // letterbox 灰边 (v5.0 默认 114)
static const float CONF_THRES = 0.25f;
static const float IOU_THRES  = 0.45f;

static const char* CLASS_NAMES[NUM_CLS] = { "pantograph", "contact point" };

// yolov5s_pantograph.yaml 里的 anchors, 单位是【原始像素】(未除 stride)
// 索引顺序 = 检测头顺序 = P3/8, P4/16, P5/32
static const float ANCHORS[3][NUM_ANCH][2] = {
    { { 10.f,  13.f}, { 16.f,  30.f}, { 33.f,  23.f} },   // P3/8  (80x60)
    { { 30.f,  61.f}, { 62.f,  45.f}, { 59.f, 119.f} },   // P4/16 (40x30)
    { {116.f,  90.f}, {156.f, 198.f}, {373.f, 326.f} },   // P5/32 (20x15)
};
static const int STRIDES[3] = { 8, 16, 32 };
// grid 尺寸, 用于识别输出头属于哪一层 (输入 640x480 -> 80x60 / 40x30 / 20x15)
static const int GRID_NY[3] = { 80, 40, 20 };
static const int GRID_NX[3] = { 60, 30, 15 };

// ----------------------------------------------------------------- 小工具
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        if (sev == Severity::kERROR || sev == Severity::kINTERNAL_ERROR)
            std::cerr << "[TRT][ERR] " << msg << std::endl;
        else if (sev == Severity::kWARNING)
            std::cerr << "[TRT][WARN] " << msg << std::endl;
    }
};
static Logger gLogger;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::cerr << "CUDA 调用失败 " #call ": "                         \
                      << cudaGetErrorString(_e) << std::endl;                \
            return false;                                                    \
        }                                                                    \
    } while (0)

static int64_t volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

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

// ----------------------------------------------------------------- 检测框
struct Detection {
    float x1, y1, x2, y2, score;
    int   cls;
};

// ----------------------------------------------------------------- 前处理
// 逐行复刻 YOLOv5-5.0 utils/datasets.py::letterbox(img, (640,480), auto=False,
// scaleup=True) —— 已与官方实现做过逐像素比对, 差异为 0
//
// 对 640x480 的真实输入图, 结果固定为: r=0.75, 上下各 pad 140 行, 左右 pad 0
// 即 43.75% 的像素是恒定的 (114,114,114) 灰边
struct LetterboxInfo { float ratio; float dw, dh; };

static cv::Mat letterbox(const cv::Mat& src, LetterboxInfo& info) {
    const int h = src.rows, w = src.cols;

    // scaleup=True => 不做 min(r, 1.0), 小图也要放大
    const float r = std::min(static_cast<float>(NET_H) / h,
                             static_cast<float>(NET_W) / w);

    const int newW = static_cast<int>(std::round(w * r));
    const int newH = static_cast<int>(std::round(h * r));

    const float dw = (NET_W - newW) / 2.f;
    const float dh = (NET_H - newH) / 2.f;

    cv::Mat resized = src;
    if (w != newW || h != newH)
        cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // 注意 ±0.1: 这是 v5.0 原版写法, 用于处理 0.5 的浮点边界
    const int top    = static_cast<int>(std::round(dh - 0.1f));
    const int bottom = static_cast<int>(std::round(dh + 0.1f));
    const int left   = static_cast<int>(std::round(dw - 0.1f));
    const int right  = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat canvas(NET_H, NET_W, CV_8UC3, cv::Scalar(PAD_VALUE, PAD_VALUE, PAD_VALUE));
    resized.copyTo(canvas(cv::Rect(left, top, resized.cols, resized.rows)));

    info.ratio = r;
    info.dw = dw;
    info.dh = dh;
    return canvas;
}

// BGR -> RGB, HWC -> CHW, /255, float32; 输出长度 = 1*3*640*480
static std::vector<float> preprocess(const cv::Mat& img, LetterboxInfo& info) {
    cv::Mat canvas = letterbox(img, info);

    std::vector<float> blob(static_cast<size_t>(3) * NET_H * NET_W);
    const size_t plane = static_cast<size_t>(NET_H) * NET_W;
    const float kInv = 1.f / 255.f;

    for (int y = 0; y < NET_H; ++y) {
        const cv::Vec3b* row = canvas.ptr<cv::Vec3b>(y);
        const size_t base = static_cast<size_t>(y) * NET_W;
        for (int x = 0; x < NET_W; ++x) {
            const cv::Vec3b& p = row[x];
            // OpenCV 是 BGR, 训练时做了 BGR->RGB, 这里顺序必须是 R,G,B
            blob[0 * plane + base + x] = static_cast<float>(p[2]) * kInv;  // R
            blob[1 * plane + base + x] = static_cast<float>(p[1]) * kInv;  // G
            blob[2 * plane + base + x] = static_cast<float>(p[0]) * kInv;  // B
        }
    }
    return blob;
}

// ----------------------------------------------------------------- 引擎封装
struct TensorSlot {
    std::string  name;
    bool         isInput = false;
    int64_t      vol     = 0;
    void*        dev     = nullptr;
    std::vector<float> host;
    std::vector<int64_t> shape;
};

class TRTEngine {
public:
    ~TRTEngine() { release(); }

    bool load(const std::string& enginePath) {
        std::ifstream f(enginePath, std::ios::binary);
        if (!f) { std::cerr << "打不开 engine: " << enginePath << std::endl; return false; }
        std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
        std::cout << "engine  : " << enginePath
                  << "  (" << blob.size() / 1024.0 / 1024.0 << " MB)" << std::endl;

        runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!runtime_) { std::cerr << "createInferRuntime 失败" << std::endl; return false; }

        // TRT 10: 两参数版本, 不再有 pluginFactory 形参
        engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
        if (!engine_) {
            std::cerr << "engine 反序列化失败 —— 最常见原因:\n"
                         "  engine 必须在【目标设备】上构建, 不能跨平台/跨 TRT 版本拷\n"
                         "  (RTX 3050 上建的 engine 不能拿到 Jetson 上用!)" << std::endl;
            return false;
        }
        ctx_ = engine_->createExecutionContext();
        if (!ctx_) { std::cerr << "createExecutionContext 失败" << std::endl; return false; }

        CUDA_CHECK(cudaStreamCreate(&stream_));

        // ---- 枚举 IO tensor (TRT 10 的 name-based 接口) ----
        const int n = engine_->getNbIOTensors();
        for (int i = 0; i < n; ++i) {
            const char* nm = engine_->getIOTensorName(i);
            TensorSlot s;
            s.name    = nm;
            s.isInput = (engine_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT);

            nvinfer1::Dims d = engine_->getTensorShape(nm);
            s.vol = volume(d);
            for (int k = 0; k < d.nbDims; ++k) s.shape.push_back(d.d[k]);

            // 动态 shape 才需要 setInputShape; 本模型是静态 640x480
            bool dynamic = false;
            for (int k = 0; k < d.nbDims; ++k) if (d.d[k] < 0) dynamic = true;
            if (dynamic && s.isInput) {
                // ⚠️ TRT 10 把 Dims4 挪进了 NvInferLegacyDims.h (legacy 头, 默认
                //    不随 NvInfer.h 引入), 所以这里手工构造 Dims, 避免依赖 legacy
                nvinfer1::Dims fixed;
                fixed.nbDims = 4;
                fixed.d[0] = 1;
                fixed.d[1] = 3;
                fixed.d[2] = NET_H;
                fixed.d[3] = NET_W;
                if (!ctx_->setInputShape(nm, fixed)) {
                    std::cerr << "setInputShape 失败: " << nm << std::endl;
                    return false;
                }
                s.shape = {1, 3, NET_H, NET_W};
                s.vol = static_cast<int64_t>(3) * NET_H * NET_W;
            }

            if (engine_->getTensorDataType(nm) != nvinfer1::DataType::kFLOAT) {
                std::cerr << "只支持 FP32 IO tensor, 但 " << nm << " 不是" << std::endl;
                return false;
            }

            s.host.assign(static_cast<size_t>(s.vol), 0.f);
            CUDA_CHECK(cudaMalloc(&s.dev, static_cast<size_t>(s.vol) * sizeof(float)));
            if (!ctx_->setTensorAddress(nm, s.dev)) {
                std::cerr << "setTensorAddress 失败: " << nm << std::endl;
                return false;
            }

            std::cout << "  " << (s.isInput ? "INPUT " : "OUTPUT") << "  "
                      << s.name << "  [";
            for (size_t k = 0; k < s.shape.size(); ++k)
                std::cout << s.shape[k] << (k + 1 < s.shape.size() ? "," : "");
            std::cout << "]" << std::endl;

            if (s.isInput) {
                inName_ = s.name;
                inVol_  = s.vol;
                if (s.shape.size() != 4 || s.shape[2] != NET_H || s.shape[3] != NET_W) {
                    std::cerr << "!! 输入形状不是 [1,3," << NET_H << "," << NET_W
                              << "], 前处理会不匹配!" << std::endl;
                }
            } else {
                outNames_.push_back(s.name);
            }

            slots_.push_back(s);
        }
        std::cout << "  engine  : " << engine_->getNbLayers() << " 层, "
                  << outNames_.size() << " 个输出头" << std::endl;

        if (outNames_.size() != 3) {
            std::cerr << "!! 期望 3 个输出头, 实得 " << outNames_.size() << std::endl;
        }
        return true;
    }

    // 返回 3 个 head 的裸输出 (未过 sigmoid), 顺序已按 P3/P4/P5 排好
    bool infer(const std::vector<float>& input, std::vector<std::vector<float>>& heads) {
        if (input.size() != static_cast<size_t>(inVol_)) {
            std::cerr << "输入长度错误: " << input.size() << " vs " << inVol_ << std::endl;
            return false;
        }
        TensorSlot* inSlot = slotPtr(inName_);
        if (!inSlot) return false;

        CUDA_CHECK(cudaMemcpyAsync(inSlot->dev, input.data(),
                                   input.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        if (!ctx_->enqueueV3(stream_)) {
            std::cerr << "enqueueV3 失败" << std::endl;
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        heads.clear();
        for (size_t i = 0; i < outNames_.size(); ++i) {
            TensorSlot* os = slotPtr(outNames_[i]);
            if (!os) return false;
            CUDA_CHECK(cudaMemcpyAsync(os->host.data(), os->dev,
                                       static_cast<size_t>(os->vol) * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // 收集并排序: 按 grid 面积降序 => P3, P4, P5
        struct OutRef { const std::vector<float>* data; int64_t area; int idx; };
        std::vector<OutRef> refs;
        for (size_t i = 0; i < outNames_.size(); ++i) {
            TensorSlot* os = slotPtr(outNames_[i]);
            int64_t area = 0;
            if (os->shape.size() == 5) area = os->shape[2] * os->shape[3];
            OutRef r{ &os->host, area, static_cast<int>(i) };
            refs.push_back(r);
        }
        std::sort(refs.begin(), refs.end(),
                  [](const OutRef& a, const OutRef& b) { return a.area > b.area; });
        for (size_t i = 0; i < refs.size(); ++i) heads.push_back(*refs[i].data);
        return true;
    }

    const std::vector<int64_t>& shapeOf(const std::string& n) {
        static const std::vector<int64_t> kEmpty;
        TensorSlot* s = slotPtr(n);
        return s ? s->shape : kEmpty;
    }
    const std::vector<std::string>& outNames() const { return outNames_; }

private:
    typedef std::vector<TensorSlot> SlotMap;
    TensorSlot* slotPtr(const std::string& n) {
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].name == n) return &slots_[i];
        return nullptr;
    }
    void release() {
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].dev) cudaFree(slots_[i].dev);
        slots_.clear();
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (ctx_)     { delete ctx_;     ctx_ = nullptr; }
        if (engine_)  { delete engine_;  engine_ = nullptr; }
        if (runtime_) { delete runtime_; runtime_ = nullptr; }
    }

    nvinfer1::IRuntime*        runtime_ = nullptr;
    nvinfer1::ICudaEngine*     engine_  = nullptr;
    nvinfer1::IExecutionContext* ctx_   = nullptr;
    cudaStream_t               stream_  = nullptr;

    std::string                inName_;
    int64_t                    inVol_ = 0;
    std::vector<std::string>   outNames_;
    SlotMap                    slots_;
};

// ----------------------------------------------------------------- 后处理
// 复刻 YOLOv5-5.0 models/yolo.py::Detect.forward 的 inference 分支
//   y[..., 0:2] = (sigmoid(xy)*2 - 0.5 + grid) * stride       -> cx, cy
//   y[..., 2:4] = (sigmoid(wh)*2)^2 * anchor_grid             -> w, h
// ⚠️ 前 4 列是 (cx,cy,w,h), 不是 xyxy! v5.0 是在 NMS 里才转的
static void decode(const std::vector<std::vector<float>>& heads,
                   std::vector<Detection>& dets)
{
    dets.clear();

    for (size_t lvl = 0; lvl < heads.size() && lvl < 3; ++lvl) {
        // head 形状 [1, 3, ny, nx, 7]
        int na = NUM_ANCH, ny = GRID_NY[lvl], nx = GRID_NX[lvl], no = NUM_NO;
        // 若能用真实形状覆盖则覆盖 (防止排序假设出错)
        const size_t expect = static_cast<size_t>(na) * ny * nx * no;
        if (heads[lvl].size() != expect) {
            std::cerr << "head " << lvl << " 尺寸异常: " << heads[lvl].size()
                      << " 期望 " << expect << " —— 请检查输出头顺序!" << std::endl;
            continue;
        }

        const float* p  = heads[lvl].data();
        const int stride = STRIDES[lvl];

        for (int a = 0; a < na; ++a) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    const float* v = p + ((static_cast<size_t>(a) * ny + y) * nx + x) * no;

                    const float obj = sigmoid_f(v[4]);
                    if (obj <= CONF_THRES) continue;

                    // 找最大类别
                    int   bestC = 0;
                    float bestP = v[5];
                    for (int c = 1; c < NUM_CLS; ++c)
                        if (v[5 + c] > bestP) { bestP = v[5 + c]; bestC = c; }

                    const float score = obj * sigmoid_f(bestP);
                    if (score <= CONF_THRES) continue;

                    const float cx = (sigmoid_f(v[0]) * 2.f - 0.5f + x) * stride;
                    const float cy = (sigmoid_f(v[1]) * 2.f - 0.5f + y) * stride;
                    const float bw = std::pow(sigmoid_f(v[2]) * 2.f, 2.f) * ANCHORS[lvl][a][0];
                    const float bh = std::pow(sigmoid_f(v[3]) * 2.f, 2.f) * ANCHORS[lvl][a][1];

                    Detection d;
                    d.x1 = cx - bw / 2.f;
                    d.y1 = cy - bh / 2.f;
                    d.x2 = cx + bw / 2.f;
                    d.y2 = cy + bh / 2.f;
                    d.score = score;
                    d.cls   = bestC;
                    dets.push_back(d);
                }
            }
        }
    }

    // 置信度降序
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    // 贪心 NMS: 同类且 IoU 超阈值才抑制
    std::vector<Detection> keep;
    std::vector<char> dead(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (dead[i]) continue;
        keep.push_back(dets[i]);
        const float a[4] = { dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2 };
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (dead[j] || dets[j].cls != dets[i].cls) continue;
            const float b[4] = { dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2 };
            if (iou_xyxy(a, b) > IOU_THRES) dead[j] = 1;
        }
    }
    dets.swap(keep);
}

// 从 letterbox 坐标映射回原图坐标
static void scaleToOriginal(std::vector<Detection>& dets, const LetterboxInfo& li,
                            int w0, int h0)
{
    for (size_t i = 0; i < dets.size(); ++i) {
        Detection& d = dets[i];
        d.x1 = (d.x1 - li.dw) / li.ratio;
        d.x2 = (d.x2 - li.dw) / li.ratio;
        d.y1 = (d.y1 - li.dh) / li.ratio;
        d.y2 = (d.y2 - li.dh) / li.ratio;
        d.x1 = std::max(0.f, std::min(d.x1, static_cast<float>(w0)));
        d.x2 = std::max(0.f, std::min(d.x2, static_cast<float>(w0)));
        d.y1 = std::max(0.f, std::min(d.y1, static_cast<float>(h0)));
        d.y2 = std::max(0.f, std::min(d.y2, static_cast<float>(h0)));
    }
}

static void draw(cv::Mat& img, const std::vector<Detection>& dets) {
    // BGR: 受电弓=红, 接触点=绿
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
                  << " <engine> <图片或目录> [输出目录]\n"
                     "例:   " << argv[0]
                  << " best_int8_minmax.engine ./val ./out\n";
        return 1;
    }
    const std::string enginePath = argv[1];
    const std::string source     = argv[2];
    const std::string outDir     = (argc > 3) ? argv[3] : "";

    TRTEngine engine;
    if (!engine.load(enginePath)) return 2;

    std::vector<cv::String> files;
    if (source.find(".jpg") != std::string::npos ||
        source.find(".png") != std::string::npos ||
        source.find(".bmp") != std::string::npos) {
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
        std::vector<float> blob = preprocess(img, li);

        std::vector<std::vector<float>> heads;
        if (!engine.infer(blob, heads)) return 4;

        std::vector<Detection> dets;
        decode(heads, dets);
        scaleToOriginal(dets, li, img.cols, img.rows);

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
            std::string base = (slash == std::string::npos) ? files[i]
                                                            : files[i].substr(slash + 1);
            cv::imwrite(outDir + "/" + base, img);
        }
    }

    std::cout << "\n合计检出 " << total << " 个目标" << std::endl;
    if (!outDir.empty()) std::cout << "标注图已存: " << outDir << std::endl;
    return 0;
}
