import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


"""
本脚本用于把高帧率 sharp 序列合成为 rolling-shutter 训练数据。
核心思想：
1) 一个 RS 样本对应一个时间窗（readout_frames）。
2) 图像每一行在该时间窗中有不同的采样时刻。
3) 若开启 row exposure，则每行在时间上做多点积分，模拟有限曝光。
"""


DEFAULT_SCENES = ["my_377", "my_386", "my_387", "my_392", "my_393", "my_394"]
DEFAULT_VIEWS = ["04", "10", "16", "22"]


def parse_args():
    """命令行参数：定义数据路径、RS 合成参数与覆盖策略。"""
    parser = argparse.ArgumentParser(
        description="Synthesize rolling-shutter training data from high-frame-rate sharp ZJU sequences."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("./data/BlurZJU/sharp"),
        help="Root directory of the rearranged sharp dataset.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("./data/BlurZJU/rs11"),
        help="Root directory of the synthesized rolling-shutter dataset.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Scene names to process.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=DEFAULT_VIEWS,
        help="Camera views to keep for training.",
    )
    parser.add_argument(
        "--readout-frames",
        type=int,
        default=11,
        help="Number of sharp frames covered by one rolling-shutter readout.",
    )
    parser.add_argument(
        "--pose-num",
        type=int,
        default=50,
        help="Number of synthesized training frames per view.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Stride between consecutive rolling-shutter windows. Defaults to readout-frames.",
    )
    parser.add_argument(
        "--row-exposure-frames",
        type=float,
        default=0.0,
        help=(
            "Temporal exposure width of each row, measured in sharp-frame units. "
            "0 means pure rolling shutter without per-row motion blur."
        ),
    )
    parser.add_argument(
        "--row-exposure-samples",
        type=int,
        default=1,
        help="Number of temporal samples used to integrate each row exposure.",
    )
    parser.add_argument(
        "--scan-direction",
        choices=["top_to_bottom", "bottom_to_top"],
        default="top_to_bottom",
        help="Sensor readout direction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output scene directory.",
    )
    return parser.parse_args()


def _read_image(path: Path) -> np.ndarray:
    """读取图像并统一为 uint8。"""
    image = imageio.imread(path)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _read_mask(path: Path) -> np.ndarray:
    """读取 mask 并归一化到 [0, 1]。"""
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.float32) / 255.0


def _write_png(path: Path, array: np.ndarray):
    """确保目录存在后写入 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, array)


def _save_smpl_json(path: Path, smpl_param: dict):
    """把中心时刻的 SMPL 参数写成与训练读取器兼容的 JSON 格式。"""
    payload = [{
        "id": 0,
        "Rh": np.asarray(smpl_param["Rh"], dtype=np.float32).tolist(),
        "Th": np.asarray(smpl_param["Th"], dtype=np.float32).tolist(),
        "poses": np.asarray(smpl_param["poses"], dtype=np.float32).tolist(),
        "shapes": np.asarray(smpl_param["shapes"], dtype=np.float32).tolist(),
    }]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _save_vertices_json(path: Path, vertices: np.ndarray):
    """把中心时刻的 SMPL 顶点写成 JSON。"""
    payload = [{
        "id": 0,
        "vertices": np.asarray(vertices, dtype=np.float32).tolist(),
    }]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _make_row_times(height: int, readout_frames: int, scan_direction: str) -> np.ndarray:
    """
    为每一行分配采样时刻：
    - 在 [0, readout_frames-1] 上按行线性分布
    - 支持从上到下 / 从下到上扫描
    """
    if height == 1:
        row_times = np.zeros((1,), dtype=np.float32)
    else:
        row_times = np.linspace(0.0, readout_frames - 1, num=height, dtype=np.float32)
    if scan_direction == "bottom_to_top":
        row_times = row_times[::-1].copy()
    return row_times


def _sample_times(row_times: np.ndarray, exposure_frames: float, samples: int) -> np.ndarray:
    """
    行曝光时间采样：
    - 若 exposure=0 或 samples=1，退化为每行单时刻采样
    - 否则在行中心时刻附近取多点，后续做平均实现时间积分
    """
    if samples <= 1 or exposure_frames <= 0:
        return row_times[:, None]

    offsets = np.linspace(
        -0.5 * exposure_frames,
        0.5 * exposure_frames,
        num=samples,
        dtype=np.float32,
    )
    return row_times[:, None] + offsets[None, :]


def _interpolate_rows(stack: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """
    沿时间维做逐行线性插值。
    输入:
    - stack: [T, H, W, C] 或 [T, H, W]
    - sample_times: [H, S]，每行 S 个采样时刻
    输出:
    - 对 S 次采样结果取均值得到最终图像/掩码
    """
    frame_count, height = stack.shape[:2]
    rows = np.arange(height, dtype=np.int64)
    outputs = []

    for sample_idx in range(sample_times.shape[1]):
        times = np.clip(sample_times[:, sample_idx], 0.0, frame_count - 1.0)
        lo = np.floor(times).astype(np.int64)
        hi = np.clip(lo + 1, 0, frame_count - 1)
        alpha = (times - lo).astype(np.float32)

        lo_rows = stack[lo, rows]
        hi_rows = stack[hi, rows]

        while alpha.ndim < lo_rows.ndim:
            alpha = alpha[..., None]

        outputs.append(lo_rows * (1.0 - alpha) + hi_rows * alpha)

    return np.mean(outputs, axis=0)


def synthesize_rs_image(
    image_stack: np.ndarray,
    mask_stack: np.ndarray,
    scan_direction: str,
    row_exposure_frames: float,
    row_exposure_samples: int,
):
    """
    合成单张 RS 图像：
    - 先计算每行采样时刻
    - 再分别对图像与 mask 做逐行时间插值
    """
    height = image_stack.shape[1]
    row_times = _make_row_times(height, image_stack.shape[0], scan_direction)
    sample_times = _sample_times(row_times, row_exposure_frames, row_exposure_samples)

    image = _interpolate_rows(image_stack.astype(np.float32), sample_times)
    mask = _interpolate_rows(mask_stack.astype(np.float32), sample_times)

    image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    mask = np.clip(np.round(mask * 255.0), 0, 255).astype(np.uint8)

    return image, mask, row_times


def load_window(scene_dir: Path, view: str, start_idx: int, readout_frames: int):
    """加载一个时间窗内的多帧图像与 mask。"""
    image_stack = []
    mask_stack = []
    for frame_idx in range(start_idx, start_idx + readout_frames):
        image_path = scene_dir / "images" / view / f"{frame_idx:06d}.jpg"
        mask_path = scene_dir / "mask" / view / f"{frame_idx:06d}.png"
        image_stack.append(_read_image(image_path))
        mask_stack.append(_read_mask(mask_path))

    return np.stack(image_stack, axis=0), np.stack(mask_stack, axis=0)


def convert_center_metadata(scene_dir: Path, out_scene_dir: Path, center_idx: int, sample_idx: int):
    """
    每个 RS 样本绑定“中心时刻”的 SMPL 参数与顶点，
    以兼容后续训练读取接口。
    """
    smpl_param = np.load(scene_dir / "smpl_params" / f"{center_idx}.npy", allow_pickle=True).item()
    vertices = np.load(scene_dir / "smpl_vertices" / f"{center_idx}.npy")

    _save_smpl_json(out_scene_dir / "smpl" / "smpl" / f"{sample_idx:06d}.json", smpl_param)
    _save_vertices_json(out_scene_dir / "smpl_vertices" / f"{sample_idx:06d}.json", vertices)


def copy_scene_static_files(scene_dir: Path, out_scene_dir: Path):
    """复制静态场景文件（如相机外参），这些内容不随 RS 合成改变。"""
    src = scene_dir / "camera_extris"
    dst = out_scene_dir / "camera_extris"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def build_metadata(
    source_scene_dir: Path,
    out_scene_dir: Path,
    view_windows: dict,
    args,
):
    """保存合成配置与每个输出样本对应的源时间窗索引。"""
    meta = {
        "source_scene": str(source_scene_dir),
        "views": list(view_windows.keys()),
        "readout_frames": args.readout_frames,
        "window_stride": args.window_stride,
        "pose_num": args.pose_num,
        "scan_direction": args.scan_direction,
        "row_exposure_frames": args.row_exposure_frames,
        "row_exposure_samples": args.row_exposure_samples,
        "samples": view_windows,
    }
    with (out_scene_dir / "rolling_shutter_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def synthesize_scene(scene: str, args):
    """对单个场景执行完整 RS 数据合成。"""
    source_scene_dir = args.source_root / scene
    out_scene_dir = args.target_root / scene

    if not source_scene_dir.exists():
        raise FileNotFoundError(f"Scene not found: {source_scene_dir}")

    if out_scene_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {out_scene_dir}. Use --force to overwrite.")
        shutil.rmtree(out_scene_dir)

    out_scene_dir.mkdir(parents=True, exist_ok=True)
    copy_scene_static_files(source_scene_dir, out_scene_dir)

    stride = args.window_stride if args.window_stride is not None else args.readout_frames
    center_offset = args.readout_frames // 2
    view_windows = {view: [] for view in args.views}

    first_view = args.views[0]
    total_frames = len(sorted((source_scene_dir / "images" / first_view).glob("*.jpg")))
    max_pose_num = (total_frames - args.readout_frames) // stride + 1
    pose_num = min(args.pose_num, max_pose_num)

    if pose_num <= 0:
        raise ValueError(
            f"Not enough frames in {source_scene_dir} for readout_frames={args.readout_frames}, stride={stride}."
        )

    progress = tqdm(range(pose_num), desc=scene)
    for sample_idx in progress:
        # 每个输出样本对应一个固定长度的时间窗
        start_idx = sample_idx * stride
        center_idx = start_idx + center_offset

        # 写入与中心时刻对应的 SMPL 元数据
        convert_center_metadata(source_scene_dir, out_scene_dir, center_idx, sample_idx)

        for view in args.views:
            image_stack, mask_stack = load_window(source_scene_dir, view, start_idx, args.readout_frames)
            rs_image, rs_mask, _ = synthesize_rs_image(
                image_stack=image_stack,
                mask_stack=mask_stack,
                scan_direction=args.scan_direction,
                row_exposure_frames=args.row_exposure_frames,
                row_exposure_samples=max(1, args.row_exposure_samples),
            )

            # 额外保存中心时刻 mask，便于训练时作为参考 GT mask 使用
            center_mask = _read_image(source_scene_dir / "mask" / view / f"{center_idx:06d}.png")

            # 输出目录结构与原 blur 数据保持一致，降低训练代码改动
            _write_png(out_scene_dir / "images" / view / f"{sample_idx:03d}.png", rs_image)
            _write_png(out_scene_dir / "mask" / view / f"{sample_idx:03d}.png", rs_mask)
            _write_png(out_scene_dir / "gt_mask" / view / f"{sample_idx:03d}.png", center_mask)

            view_windows[view].append({
                "sample_idx": sample_idx,
                "start_idx": start_idx,
                "center_idx": center_idx,
                "end_idx": start_idx + args.readout_frames - 1,
                "output_image": f"images/{view}/{sample_idx:03d}.png",
            })

    build_metadata(source_scene_dir, out_scene_dir, view_windows, args)


def main():
    """参数检查与批量场景处理入口。"""
    args = parse_args()

    if args.readout_frames <= 0:
        raise ValueError("--readout-frames must be positive.")
    if args.pose_num <= 0:
        raise ValueError("--pose-num must be positive.")
    if args.row_exposure_samples <= 0:
        raise ValueError("--row-exposure-samples must be positive.")

    if args.window_stride is None:
        args.window_stride = args.readout_frames
    if args.window_stride <= 0:
        raise ValueError("--window-stride must be positive.")

    # 统一视角名为两位字符串（如 4 -> "04"）
    args.views = [f"{int(view):02d}" for view in args.views]

    args.target_root.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        synthesize_scene(scene, args)


if __name__ == "__main__":
    main()
