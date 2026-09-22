#!/usr/bin/env bash
# ============================================================================
#  run_check.sh —— 在没有 GPU / 没有 OpenCV 开发包 / 没有 rknpu2 的机器上,
#                  对 deploy 下的 C++ 做真实的「解析 + 代码生成」检查。
#
#  原理: 用 zig 自带 clang 作为编译器, 配合 _syntax_check/include 下的【桩头文件】
#        (OpenCV / CUDA Runtime / RKNN) 补上缺失的第三方声明。
#        TensorRT 那份用的是【真实】的 NvInfer.h, 所以 TRT API 调用能被真校验。
#
#  能查出: 语法错误、类型错误、函数签名不匹配、成员不存在, 以及 TensorRT 10 的
#          API 用法错误 (这是最容易踩的地方)
#  查不出: OpenCV / CUDA / RKNN 这些被 stub 掉的 API 的真实签名是否正确, 以及
#          链接期问题 —— 上真机时仍需在目标平台本地编译一次
#
#  用法:
#     bash deploy/_syntax_check/run_check.sh
#     TRT_ROOT=/path/to/TensorRT-10.0.1.6 bash deploy/_syntax_check/run_check.sh
#     ZIG=/path/to/zig bash deploy/_syntax_check/run_check.sh
#
#  注: 本机 Git Bash 的 PATH 是坏的 (dirname/mktemp 都可能 command not found),
#      跑之前先 export PATH="/usr/bin:/bin:$PATH"
# ============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
DEPLOY="$(cd "$HERE/.." && pwd)"
INC="$HERE/include"

# zig.exe 是 Windows 程序, 喂给它 POSIX 风格路径 (/f/...) 会得到
# FileNotFound / CacheCheckFailed, 所以统一转成 F:/... 这种形式
winpath() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$1"
    else
        printf '%s' "$1" | sed -E 's|^/([a-zA-Z])/|\1:/|'
    fi
}
W_DEPLOY="$(winpath "$DEPLOY")"
W_INC="$(winpath "$INC")"

# 工作目录放在脚本旁边 (必须是 zig 认得的路径, 不能用 /tmp)
WORK="$HERE/.tmp_check"
rm -rf "$WORK"; mkdir -p "$WORK"
W_WORK="$(winpath "$WORK")"

# 被 SIGTERM 打断过的 zig 会把默认缓存写坏 -> FileNotFound / CacheCheckFailed。
# 显式指定缓存目录, 收尾时删掉, 不污染用户环境
export ZIG_GLOBAL_CACHE_DIR="$W_WORK/global"
export ZIG_LOCAL_CACHE_DIR="$W_WORK/local"

# ---- 找 zig ---------------------------------------------------------------
ZIG="${ZIG:-}"
if [ -z "$ZIG" ] && command -v zig >/dev/null 2>&1; then
    ZIG="$(command -v zig)"
fi
if [ -z "$ZIG" ]; then
    for PY in \
        "${ZIGLANG_PYTHON:-}" \
        "$(command -v python 2>/dev/null)" \
        "$(command -v python3 2>/dev/null)" \
        "C:/Users/33048/.workbuddy/binaries/python/envs/yoloint8/Scripts/python.exe"
    do
        [ -z "$PY" ] && continue
        [ -x "$PY" ] || continue
        DIR="$("$PY" -c 'import ziglang,os;print(os.path.dirname(ziglang.__file__))' 2>/dev/null)"
        [ -z "$DIR" ] && continue
        DIR="$(printf '%s' "$DIR" | tr '\\' '/')"
        for exe in zig.exe zig; do
            if [ -x "$DIR/$exe" ]; then ZIG="$DIR/$exe"; break 2; fi
        done
    done
fi
if [ -z "$ZIG" ]; then
    echo "!! 找不到 zig。装一个:  pip install ziglang"
    echo "   或手动指定:  ZIG=/path/to/zig bash $0"
    exit 1
fi
echo "编译器 : $ZIG"
echo "桩头   : $W_INC"

COMMON="-std=c++14 -O1 -Wno-nullability-completeness"
FAILED=0

run_case() {          # $1=标题 $2=源文件(win) $3=额外 -I $4=tag
    echo
    echo "=============================================================="
    echo "  $1"
    echo "=============================================================="
    LOG="$W_WORK/$4.log"
    # shellcheck disable=SC2086
    "$ZIG" c++ -c -o "$W_WORK/$4.o" $COMMON -I "$W_INC" $3 "$2" > "$LOG" 2>&1
    local rc=$?
    # 只显示来自我们自己文件的诊断, 过滤 libcxx 的噪音
    grep -E "$(basename "$2"):[0-9]+:[0-9]+: (error|warning)" "$LOG" || true
    if [ $rc -eq 0 ]; then
        local sz; sz=$(stat -c%s "$W_WORK/$4.o" 2>/dev/null || echo "?")
        echo "  >>> PASS  (exit=0, 产码 $sz 字节)"
        return 0
    fi
    echo "  >>> FAIL  (exit=$rc)"
    grep -E "error" "$LOG" | head -20
    return 1
}

TRT_ROOT="${TRT_ROOT:-D:/Enviroment/TensorRT-10.0.1.6}"
if [ -f "$TRT_ROOT/include/NvInfer.h" ]; then
    run_case "[1/2] deploy/jetson/yolo_trt.cpp   (真实 NvInfer.h)" \
             "$W_DEPLOY/jetson/yolo_trt.cpp" "-I $TRT_ROOT/include" jetson \
        || FAILED=1
else
    echo
    echo "!! 在 $TRT_ROOT 下找不到 NvInfer.h, 跳过 Jetson 用例"
    echo "   (用 TRT_ROOT=/path/to/TensorRT-x.x 指定)"
fi

run_case "[2/2] deploy/rk3588/rknn_yolo.cpp  (stub rknn_api.h)" \
         "$W_DEPLOY/rk3588/rknn_yolo.cpp" "" rknn \
    || FAILED=1

echo
if [ $FAILED -eq 0 ]; then
    echo "两条链路均通过编译期检查。"
    rm -rf "$WORK"
    exit 0
else
    echo "有失败项, 日志保留在: $WORK"
    exit 1
fi
