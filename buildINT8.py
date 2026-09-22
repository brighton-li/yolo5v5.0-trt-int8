# -*- coding: utf-8 -*-
"""
buildINT8.py —— 用 TensorRT 10.0.1 构建 FP16 / INT8 引擎

原脚本有个问题:`builder.build_engine()` 在 TensorRT 10 已被【移除】, 必须改用


用法:
    python buildINT8.py --mode fp16            # 只建 FP16
    python buildINT8.py --mode int8            # 只建 INT8 (PTQ + 500张校准)
    python buildINT8.py --mode both            # 都建 (默认)

注意: TensorRT 的 DLL 路径由 trt_env.py 负责注入, 不要靠 shell 的 PATH
      (Git Bash 会把 D:/... 形式的 PATH 搞坏)
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trt_env  # noqa: F401  —— 必须在 import tensorrt 之前

import tensorrt as trt

ONNX_DEFAULT = os.path.join(HERE, "best.onnx")
CALIB_DEFAULT = os.path.join(HERE, "calib_500.txt")


def make_network(builder, logger):
    """TensorRT 10 里 EXPLICIT_BATCH 已默认开启; 兼容老写法"""
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        return builder.create_network(flag)
    except Exception:
        return builder.create_network()


def build_engine(onnx_path, engine_path, mode, calib_list=None,
                 cache_file=None, workspace_gb=1.0, batch_size=10,
                 calib_algo="entropy", verbose=True):
    assert mode in ("fp16", "int8")
    logger = trt.Logger(trt.Logger.WARNING if not verbose else trt.Logger.INFO)

    builder = trt.Builder(logger)
    network = make_network(builder, logger)
    parser = trt.OnnxParser(network, logger)

    print(f"\n{'-'*74}")
    print(f"  构建 [{mode.upper()}] 引擎")
    print(f"{'-'*74}")
    print(f"  ONNX      : {onnx_path}  ({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

    t0 = time.time()
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        print("  !! ONNX 解析失败:")
        for i in range(parser.num_errors):
            print("    ", parser.get_error(i))
        raise RuntimeError("ONNX 解析失败")
    print(f"  解析成功  : {network.num_layers} 层, "
          f"{network.num_inputs} 输入, {network.num_outputs} 输出  "
          f"({time.time()-t0:.2f}s)")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"    in  {t.name:<10} shape={tuple(t.shape)} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"    out {t.name:<10} shape={tuple(t.shape)} {t.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * (1 << 30)))
    try:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    except Exception:
        pass

    if mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("  精度策略  : FP16 (允许 FP16 算子, 其余回退 FP32)")
    else:
        # INT8 + FP16: 标准做法。无法量化/量化收益为负的层回退 FP16/FP32
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_algo == "minmax":
            from calibrate import MyMinMaxCalibrator as Cal
            algo_name = "MINMAX_CALIBRATION"
        else:
            from calibrate import MyCalibrator as Cal
            algo_name = "ENTROPY_CALIBRATION_2"
        cal = Cal(calib_list, batch_size=batch_size, cache_file=cache_file)
        config.int8_calibrator = cal
        print(f"  精度策略  : INT8 + FP16 回退")
        print(f"  校准算法  : {algo_name}")
        print(f"  校准集    : {calib_list}  ({len(cal.paths)} 张, batch={batch_size})")
        print(f"  校准缓存  : {cache_file}")

    print(f"  workspace : {workspace_gb:.1f} GB")
    print("  开始构建  ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine 构建失败 (build_serialized_network 返回 None)")
    blob = bytes(serialized)
    with open(engine_path, "wb") as f:
        f.write(blob)
    print(f"  构建完成  : {time.time()-t0:.1f}s -> {engine_path}")
    print(f"  engine 体积: {len(blob)/1024/1024:.2f} MB")

    # 反序列化核对 IO
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError("engine 反序列化失败")
    print(f"  引擎核对  : {engine.num_io_tensors} 个 IO tensor, "
          f"{engine.num_layers} 层")
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        print(f"    {engine.get_tensor_mode(n).name:<8} {n:<10} "
              f"shape={tuple(engine.get_tensor_shape(n))} "
              f"{engine.get_tensor_dtype(n)}")

    # 逐层精度报告 (看哪些层没吃上 INT8)
    try:
        if mode == "int8":
            insp = engine.create_engine_inspector()
            info = insp.get_engine_information(
                trt.LayerInformationFormat.JSON)
            p = os.path.splitext(engine_path)[0] + "_layers.json"
            with open(p, "w", encoding="utf-8") as f:
                f.write(info)
            cnt = {}
            for key in ('"Type": "Int8"', '"Type":"Int8"',
                        '"Type": "Float"', '"Type":"Float"',
                        '"Type": "Half"', '"Type":"Half"'):
                c = info.count(key)
                if c:
                    cnt[key.split(":")[1].strip().strip('"')] = \
                        cnt.get(key.split(":")[1].strip().strip('"'), 0) + c
            print(f"  逐层精度统计 (tensor 类型出现次数): {cnt}")
            print(f"  详细逐层报告: {p}")
    except Exception as e:
        print(f"  (逐层精度报告跳过: {type(e).__name__}: {e})")

    del engine, runtime, serialized
    return engine_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=ONNX_DEFAULT)
    ap.add_argument("--mode", default="both", choices=["fp16", "int8", "both"])
    ap.add_argument("--calib", default=CALIB_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=10,
                    help="校准 batch (必须整除校准图数, 500/10=50)")
    ap.add_argument("--calib-algo", default="entropy", choices=["entropy", "minmax"])
    ap.add_argument("--tag", default="", help="输出文件名后缀, 如 _minmax")
    ap.add_argument("--workspace-gb", type=float, default=1.0)
    ap.add_argument("--outdir", default=HERE)
    opt = ap.parse_args()

    print("=" * 74)
    print("  TensorRT 引擎构建")
    print("=" * 74)
    print(f"  TensorRT 版本 : {trt.__version__}")
    print(f"  CUDA 可用     : {__import__('torch').cuda.is_available()}")
    print(f"  GPU           : {__import__('torch').cuda.get_device_name(0)}")

    results = []
    if opt.mode in ("fp16", "both"):
        results.append(build_engine(
            opt.onnx, os.path.join(opt.outdir, "best_fp16.engine"),
            "fp16", workspace_gb=opt.workspace_gb))
    if opt.mode in ("int8", "both"):
        tag = opt.tag or ("" if opt.calib_algo == "entropy" else "_minmax")
        results.append(build_engine(
            opt.onnx, os.path.join(opt.outdir, f"best_int8{tag}.engine"),
            "int8",
            calib_list=opt.calib,
            cache_file=os.path.join(opt.outdir, f"best_int8{tag}.cache"),
            batch_size=opt.batch_size,
            calib_algo=opt.calib_algo,
            workspace_gb=opt.workspace_gb))

    print("\n" + "=" * 74)
    print("  全部完成:")
    for r in results:
        print(f"    {r}  ({os.path.getsize(r)/1024/1024:.2f} MB)")
    print("=" * 74)
