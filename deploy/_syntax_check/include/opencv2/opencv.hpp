// ============================================================================
//  opencv2/opencv.hpp —— 【仅用于语法检查的桩头文件】
//
//  用途: 本机没有 OpenCV 的 C++ 开发包, 为了用 zig/clang 对 deploy/*.cpp 做
//        -fsyntax-only 检查, 这里声明被用到的那些 OpenCV 符号。
//        【不参与真正的编译, 不要拿去构建!】
// ============================================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cv {

typedef std::string String;

class Vec3b {
public:
    unsigned char v[3];
    unsigned char& operator[](int i) { return v[i]; }
    const unsigned char& operator[](int i) const { return v[i]; }
};

class Point {
public:
    int x, y;
    Point(int _x = 0, int _y = 0) : x(_x), y(_y) {}
};

class Size {
public:
    int width, height;
    Size(int w = 0, int h = 0) : width(w), height(h) {}
};

class Rect {
public:
    int x, y, width, height;
    Rect(int _x = 0, int _y = 0, int _w = 0, int _h = 0)
        : x(_x), y(_y), width(_w), height(_h) {}
};

class Scalar {
public:
    double v[4];
    Scalar(double a = 0, double b = 0, double c = 0, double d = 0) {
        v[0] = a; v[1] = b; v[2] = c; v[3] = d;
    }
};

class Mat {
public:
    int rows = 0, cols = 0;
    Mat() {}
    Mat(int r, int c, int type, const Scalar& s) { (void)type; (void)s; rows = r; cols = c; }
    bool empty() const { return rows == 0 || cols == 0; }
    template <typename T> T* ptr(int) { return nullptr; }
    template <typename T> const T* ptr(int) const { return nullptr; }
    void copyTo(const Mat& _dst) const { (void)_dst; }
    Mat operator()(const Rect&) const { return Mat(); }
};

// 真实 OpenCV 里 CV_8UC3 是【宏】(CV_MAKETYPE 拼出来的), 不是命名空间内的枚举,
// 所以这里也用宏, 才能如实反映「不加 cv:: 前缀能不能用」
#define CV_8UC3 16

enum {
    INTER_LINEAR = 1,
    FONT_HERSHEY_SIMPLEX = 0
};

Mat imread(const String& filename, int flags = 1);
void resize(const Mat& src, Mat& dst, const Size& dsize,
            double fx = 0, double fy = 0, int interpolation = 1);
void rectangle(Mat& img, const Point& p1, const Point& p2, const Scalar& color,
               int thickness = 1, int lineType = 8, int shift = 0);
void putText(Mat& img, const String& text, const Point& org, int fontFace,
             double fontScale, const Scalar& color, int thickness = 1,
             int lineType = 8, bool bottomLeftOrigin = false);
Size getTextSize(const String& text, int fontFace, double fontScale,
                 int thickness, int* baseLine);
bool imwrite(const String& filename, const Mat& img,
             const std::vector<int>& params = std::vector<int>());
void glob(const String& pattern, std::vector<String>& result, bool recursive = false);

}  // namespace cv
