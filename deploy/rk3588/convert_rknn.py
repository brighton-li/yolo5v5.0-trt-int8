# -*- coding: utf-8 -*-
"""
convert_rknn.py —— ONNX -> RKNN (RK3588 NPU)

 这个脚本必须在 【x86_64 Linux】 上跑, 不能在 RK3588 板子上跑。
   rknn-toolkit2 只有 x86_64 Linux 版本 (Windows / ARM 都没有)。

建议 conda 新建 py3.8~3.10 环境:
    pip install rknn-toolkit2 -i https://pypi.tuna.tsinghua.edu.cn/simple
    # 或从 RK 官方 repo 装 wheel:
    #   https://github.com/airockchip/rknn-toolkit2  ->  rknn_toolkit2-*.whl

板端运行时 (不同包, 不要搞混):
    rknpu2 里的 librknnrt.so + rknn_api.h  ->  C++ 编译时链接

用法:
    python convert_rknn.py --onnx best.onnx --dataset calib_500.txt \
                           --out best_rk3588.rknn --platform rk3588

"""
import os
import sys
import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="best.onnx")
    ap.add_argument("--dataset", default="calib_500.txt",
                    help="校准图清单 (每行一个绝对路径), 直接复用 TensorRT 那份")
    ap.add_argument("--out", default="best_rk3588.rknn")
    ap.add_argument("--platform", default="rk3588",
                    choices=["rk3588", "rk3588s", "rk3568", "rk3566"])
    ap.add_argument("--no-quant", action="store_true",
                    help="只转 FP16, 不做 INT8 量化 (精度对照用)")
    return ap.parse_args()


def main():
    opt = parse_args()

    try:
        from rknn.api import RKNN
    except ImportError:
        print("!! 没有 rknn-toolkit2。请在 x86_64 Linux 上:\n"
              "     pip install rknn-toolkit2\n"
              "   注意本脚本不能在 Windows / RK3588 板上运行。")
        return 1

    if not os.path.exists(opt.onnx):
        print(f"!! 找不到 ONNX: {opt.onnx}")
        return 1

    n_calib = 0
    if not opt.no_quant:
        if not os.path.exists(opt.dataset):
            print(f"!! 找不到校准清单: {opt.dataset}")
            return 1
        with open(opt.dataset, "r", encoding="utf-8-sig") as f:
            lines = [l.strip() for l in f if l.strip()]
        missing = [p for p in lines if not os.path.exists(p)]
        if missing:
            print(f"!! 校准清单里有 {len(missing)} 张图不存在, 例如 {missing[:2]}")
            print("   请确认图片路径在【这台 x86 Linux 机器】上也能访问 "
                  "(必要时先把训练图拷过来)")
            return 1
        n_calib = len(lines)

    rknn = RKNN(verbose=True)

    # ---------------------------------------------------------- 1. config
    rknn.config(
        # 输入是 RGB(已由 RKNN 按下面顺序做归一化), 与训练口径一致
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],       # /255 归一化到 0~1
        # 量化类型: 非对称 8bit (RK3588 NPU 最成熟的路径)
        quantized_dtype="asymmetric_quantized-8",
        # 量化算法: normal 就是 MinMax —— 与 TensorRT 上胜出的方案一致
        quantized_algorithm="normal",
        # 量化方法: 默认 per-channel, 对卷积更友好
        quantized_method="channel",
        target_platform=opt.platform,
        optimization_level=3,
        # ⚠️ 不要设 force_builtin_perm=True, YOLOv5 的 head 需要保留原始重排
    )

    # ---------------------------------------------------------- 2. load
    print(f"\n>>> load_onnx: {opt.onnx}")
    if rknn.load_onnx(model=opt.onnx) != 0:
        print("!! load_onnx 失败。常见原因与对策:\n"
              "   * 输出是 5D 张量 (1,3,ny,nx,7) -> RKNN 对 5D 支持很差。\n"
              "     对策: 导出 ONNX 时把 head 输出改成 3D (见 README 的\n"
              "     『5D 输出头问题』一节), 或直接把 (na,ny,nx) 展平成一条。\n"
              "   * 含 NMS / 自定义算子 -> best.onnx 里没有 NMS, 应该不是这个。")
        return 1

    # ---------------------------------------------------------- 3. build
    do_quant = (not opt.no_quant)
    print(f"\n>>> build  (do_quantization={do_quant}, "
          f"校准集={opt.dataset if do_quant else '无'} "
          f"{'(' + str(n_calib) + ' 张)' if do_quant else ''})")
    if rknn.build(do_quantization=do_quant,
                  dataset=opt.dataset if do_quant else None) != 0:
        print("!! build 失败")
        return 1

    # ---------------------------------------------------------- 4. export
    print(f"\n>>> export_rknn: {opt.out}")
    if rknn.export_rknn(opt.out) != 0:
        print("!! export_rknn 失败")
        return 1

    # ---------------------------------------------------------- 5. 报告
    size_mb = os.path.getsize(opt.out) / 1024 / 1024
    print("\n" + "=" * 74)
    print(f"  转换成功: {opt.out}  ({size_mb:.2f} MB)")
    print("=" * 74)
    print("  对照参考 (同一模型的 TensorRT 产物):")
    print("    TRT-FP16        16.16 MB   mAP@0.5 = 0.9916")
    print("    TRT-INT8-MinMax  9.75 MB   mAP@0.5 = 0.9910")
    print(f"    RKNN-INT8        {size_mb:.2f} MB   mAP 待板端实测")
    print()
    print("  下一步: 把 .rknn 拷到板子, 用 rknn_yolo.cpp 推理")
    print("          ./rknn_yolo best_rk3588.rknn test.jpg out.jpg")

    rknn.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
