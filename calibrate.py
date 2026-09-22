# -*- coding: utf-8 -*-
"""
calibrate.py —— YOLOv5-5.0 受电弓检测模型的 INT8 熵校准器 (TensorRT PTQ)

【为什么前处理必须这样写】
本模型训练时的配置 (runs/train/exp19/opt.yaml):
    img_size: [640, 480]      # 注意 v5.0 里这个顺序是 (height, width)
    rect: false
    cfg: models/yolov5s_pantograph.yaml   # nc=2
所以训练时 datasets.py 走的是:
    letterbox(img, (640, 480), auto=False, scaleup=True)
即: 实际图 640(宽)x480(高) 被缩放到 480x360, 上下各补 140 行灰边 (114),
最终得到 **宽480 x 高640** 的输入 —— 也就是 ONNX 里的 [1,3,640,480]。
校准器的前处理必须与此逐位一致, 否则校准出来的激活直方图分布是错的。

【接口】
TensorRT 10.0.1 的 Python 绑定里, IInt8EntropyCalibrator2 仍然可用:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = MyCalibrator("calib_500.txt")
"""
import os
import sys
import numpy as np
import cv2

# ---- 必须在 import tensorrt 之前加载 TensorRT 的 DLL 搜索路径 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_env  # noqa: F401

import tensorrt as trt
import torch

# ---------------------------------------------------------------- 常量
H, W = 640, 480                      # 网络输入 高x宽
COLOR = (114, 114, 114)              # YOLOv5 letterbox 填充灰
MEAN_NONE = True                     # v5.0 不做减均值, 只 /255


def letterbox_v5(img, new_shape=(H, W), color=COLOR, auto=False,
                 scaleFill=False, scaleup=True, stride=32):
    """逐行复刻 YOLOv5-5.0 utils/datasets.py::letterbox 的行为。

    auto=False -> 不做 stride 对齐 (保证输出严格等于 new_shape)
    scaleup=True -> 小图也放大 (训练时 augment=True 的取值)
    """
    shape = img.shape[:2]            # 当前 (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))   # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def preprocess(path_or_img):
    """BGR 读图 -> letterbox(640,480) -> BGR2RGB -> CHW -> float32 /255

    返回 (1, 3, 640, 480) 的 float32 ndarray
    """
    img = cv2.imread(path_or_img) if isinstance(path_or_img, str) else path_or_img
    if img is None:
        raise FileNotFoundError(f"读不到图片: {path_or_img}")
    img = letterbox_v5(img, (H, W), auto=False, scaleup=True)[0]   # (640,480,3)
    img = img[:, :, ::-1].transpose(2, 0, 1)                       # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)
    return img[None].astype(np.float32) / 255.0


class _CalibratorBase:
    """两种校准算法共用的全部逻辑 (不继承任何 trt 类, 只做 mixin)"""

    def setup(self, calib_list, batch_size=10, cache_file=None,
              input_shape=(3, H, W)):
        with open(calib_list, "r", encoding="utf-8-sig") as f:
            self.paths = [l.strip() for l in f if l.strip()]
        if not self.paths:
            raise RuntimeError(f"校准清单为空: {calib_list}")

        missing = [p for p in self.paths[:50] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"校准清单里有不存在的图片, 例如: {missing[:3]}\n"
                f"共检查前 50 条, 缺失 {len(missing)} 条")

        self.batch_size = int(batch_size)
        self.shape = (self.batch_size,) + tuple(input_shape)
        self.cache_file = cache_file
        self.cursor = 0
        self.batch_count = 0

        # 常驻显存缓冲, 只分配一次, 避免 TRT 读取期间被 allocator 复用
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用, INT8 校准需要 GPU")
        self.device = torch.device("cuda:0")
        self.host_buf = torch.zeros(self.shape, dtype=torch.float32)
        self.dev_buf = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.dev_ptr = int(self.dev_buf.data_ptr())

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """TRT 反复调用本方法直到返回 None"""
        if self.cursor >= len(self.paths):
            print(f"    [calibrator] 校准结束, 共喂入 {self.batch_count} 个 batch "
                  f"({self.cursor} 张图)")
            return None

        idx = list(range(self.cursor, min(self.cursor + self.batch_size, len(self.paths))))
        while len(idx) < self.batch_size:            # 不足一批时补齐, TRT 只认 batch_size
            idx.append(idx[len(idx) - 1])

        batch = np.empty(self.shape, dtype=np.float32)
        for k, i in enumerate(idx):
            batch[k] = preprocess(self.paths[i])[0]

        self.host_buf.copy_(torch.from_numpy(batch))
        self.dev_buf.copy_(self.host_buf)
        torch.cuda.synchronize()

        self.cursor += self.batch_size
        self.batch_count += 1
        if self.batch_count % 10 == 0 or self.batch_count == 1:
            print(f"    [calibrator] batch {self.batch_count:>3}  "
                  f"已用 {min(self.cursor, len(self.paths))}/{len(self.paths)} 张")

        return [self.dev_ptr]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"    [calibrator] 命中校准缓存: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"    [calibrator] 校准缓存已写出: {self.cache_file} "
                  f"({len(cache)/1024:.1f} KB)")


class MyCalibrator(_CalibratorBase, trt.IInt8EntropyCalibrator2):
    """INT8 熵校准器 (Entropy Calibration 2) —— TensorRT 默认推荐

    参数
    ----
    calib_list : str   校准图清单文件, 每行一个绝对路径
    batch_size : int   每个校准 batch 的图数 (建议整除图片总数)
    cache_file : str   校准缓存 .cache 路径, 命中则跳过重跑
    input_shape: tuple (C, H, W)
    """

    def __init__(self, calib_list, batch_size=10, cache_file=None,
                 input_shape=(3, H, W)):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.setup(calib_list, batch_size, cache_file, input_shape)

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2


class MyMinMaxCalibrator(_CalibratorBase, trt.IInt8MinMaxCalibrator):
    """INT8 MinMax 校准器 —— 对检测头/小目标有时比 Entropy 更稳

    注意: 不要重写 get_algorithm(), 基类会返回 MINMAX_CALIBRATION
    """

    def __init__(self, calib_list, batch_size=10, cache_file=None,
                 input_shape=(3, H, W)):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.setup(calib_list, batch_size, cache_file, input_shape)



# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    print("=" * 70)
    print("  calibrate.py 自检")
    print("=" * 70)

    ROOT = r"F:\Codes\yolov5-5.0-pantograph"
    sys.path.insert(0, ROOT)
    from utils.datasets import letterbox as ref_letterbox

    img_path = r"F:\Codes\yolov5-5.0-pantograph\VOCdevkit\images\train\00000000.jpg"
    img0 = cv2.imread(img_path)
    print(f"\n原图 shape (h,w,c): {img0.shape}")

    mine = letterbox_v5(img0, (H, W), auto=False, scaleup=True)[0]
    ref = ref_letterbox(img0, (H, W), auto=False, scaleup=True)[0]
    print(f"本实现 letterbox_out : {mine.shape}")
    print(f"仓库 letterbox_out   : {ref.shape}")
    same = np.array_equal(mine, ref)
    print(f"与 YOLOv5-5.0 官方 letterbox 逐像素一致: {same}   "
          f"最大差异 {int(np.max(np.abs(mine.astype(int)-ref.astype(int))))}")
    assert same, "letterbox 与官方实现不一致!"

    x = preprocess(img_path)
    print(f"\n网络输入张量: shape={x.shape} dtype={x.dtype} "
          f"min={x.min():.4f} max={x.max():.4f}")
    print(f"灰边占位值 (归一化后): {114/255:.4f}")
    row_center = x[0, :, 320, :].mean()
    row_pad = x[0, :, 5, :].mean()
    print(f"第 320 行(图像区)均值={row_center:.4f}, 第 5 行(灰边区)均值={row_pad:.4f}"
          f"  -> 灰边区应≈{114/255:.4f}")

    # 校准器构造测试
    CAL = r"F:\DocumentSoftFiles\workbuddy_files\yoloint8\calib_500.txt"
    if os.path.exists(CAL):
        c = MyCalibrator(CAL, batch_size=10, cache_file="test.cache")
        print(f"\nMyCalibrator 构造成功: {len(c.paths)} 张图, batch={c.get_batch_size()}, "
              f"algo={c.get_algorithm()}")
        b = c.get_batch(["images"])
        print(f"get_batch 返回设备指针: {b[0]} (torch device ptr={c.dev_buf.data_ptr()})")
        print(f"指针一致: {b[0] == c.dev_buf.data_ptr()}")
        # 再取一个 batch, 指针应保持不变
        b2 = c.get_batch(["images"])
        print(f"第二个 batch 指针不变: {b2[0] == b[0]}")
    else:
        print(f"\n(跳过校准器测试, 未找到 {CAL})")

    print("\n自检完成")
