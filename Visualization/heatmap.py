#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYSU-MM01 特征响应 Heatmap + 矩形遮挡可视化

输出四行论文风格图：
(a) Original images
(b) Feature-response heatmaps
(c) Occluded images
(d) Heatmaps of occluded images

说明：
1. 这不是 Grad-CAM，而是目标层特征的 channel-wise response map：
       heatmap = mean(feature^2, channel)
2. 默认 hook 最后一层卷积特征：
       base_resnet.base.layer4
3. 若想看 MSDEE / DG-CLRM 的响应，只需修改 TARGET_LAYER
4. 脚本一次只加载并运行一个模型。
"""

from __future__ import print_function

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import embed_net


# ============================================================
# 1. 主要配置：通常只需要修改这一部分
# ============================================================
MODEL_DISPLAY_NAME = "MDGRN"

SYSU_PATH = "./Datasets/SYSU-MM01/"

MODEL_PATH = "save_model"
RESUME = ""

# 默认观察共享 backbone 的最后一个空间特征层。
# 若想展示特定模块，请改为 model.py 中真实的模块路径。
TARGET_LAYER = "base_resnet.base.layer4"

DEVICE = "cuda:0"
N_CLASS = 395
ARCH = "resnet50"

IMG_W = 144
IMG_H = 384

# 固定身份。脚本会为每个身份各找 1 张 Visible 和 1 张 Infrared 图像。
# 论文图列数 = 2 * len(SELECTED_IDS)。
SELECTED_IDS = [89, 90, 93, 102, 105]
IMAGE_INDEX = 0

VIS_CAMS = ("cam1", "cam2", "cam4", "cam5")
IR_CAMS = ("cam3", "cam6")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# 若希望完全固定具体图片，可直接填写绝对路径。
# 一旦某个列表非空，就优先使用这里的路径。
MANUAL_VISIBLE_PATHS: List[str] = []
MANUAL_INFRARED_PATHS: List[str] = []

OUTPUT_ROOT = "save_heatmap_visualization"
OUTPUT_BASENAME = "sysu_mdgrn_heatmap_occlusion"

# Heatmap 参数
RESPONSE_MODE = "square_mean"   # square_mean / abs_mean / max_abs
COLORMAP = "jet"
OVERLAY_ALPHA = 0.45
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0

# 遮挡设置
SEED = 0
OCCLUSION_WIDTH_RATIO = 0.36
OCCLUSION_HEIGHT_RATIO = 0.32
OCCLUSION_COLOR = (145, 138, 122)
RANDOM_OCCLUSION_POSITION = True

# 绘图参数
CELL_WIDTH = 0.72
CELL_HEIGHT = 1.76
WSPACE = 0.035
HSPACE = 0.040
SAVE_DPI = 600


# ============================================================
# 2. 命令行参数
# ============================================================
parser = argparse.ArgumentParser(
    description="SYSU-MM01 feature-response heatmap visualization"
)
parser.add_argument("--arch", default=ARCH, type=str)
parser.add_argument("--resume", default=RESUME, type=str)
parser.add_argument("--model_path", default=MODEL_PATH, type=str)
parser.add_argument("--dataset_root", default=SYSU_PATH, type=str)
parser.add_argument("--target_layer", default=TARGET_LAYER, type=str)
parser.add_argument("--gpu", default=DEVICE, type=str)
parser.add_argument("--output_root", default=OUTPUT_ROOT, type=str)
args = parser.parse_args()


# ============================================================
# 3. 兼容工具
# ============================================================
try:
    PIL_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    PIL_BILINEAR = Image.BILINEAR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA 不可用，自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_name)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def build_model(device: torch.device):
    constructors = [
        lambda: embed_net(N_CLASS, dataset="sysu", arch=args.arch),
        lambda: embed_net(N_CLASS, "sysu", arch=args.arch),
        lambda: embed_net(N_CLASS, gm_pool="off", arch=args.arch),
        lambda: embed_net(N_CLASS, arch=args.arch),
    ]

    net = None
    constructor_errors = []

    for constructor in constructors:
        try:
            net = constructor()
            break
        except TypeError as error:
            constructor_errors.append(str(error))

    if net is None:
        raise TypeError(
            "无法调用 embed_net 构造函数，请根据 model.py 修改 build_model()。\n"
            + "\n".join(constructor_errors)
        )

    checkpoint_path = args.resume
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(args.model_path, checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"未找到 checkpoint：{checkpoint_path}")

    print("Loading model:")
    print("  name       :", MODEL_DISPLAY_NAME)
    print("  checkpoint :", checkpoint_path)
    print("  target     :", args.target_layer)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint 中未找到有效 state_dict。")

    state_dict = strip_module_prefix(state_dict)
    net.load_state_dict(state_dict, strict=True)

    net = net.to(device)
    net.eval()

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("  epoch      :", checkpoint["epoch"])

    return net


# ============================================================
# 4. 数据预处理与图像选择
# ============================================================
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

transform_test = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    normalize,
])


def list_images(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []

    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(IMAGE_EXTENSIONS)
    ]


def find_image_for_id(
    dataset_root: str,
    pid: int,
    cameras: Sequence[str],
    image_index: int,
) -> Optional[str]:
    candidates: List[str] = []

    for camera in cameras:
        identity_dir = os.path.join(dataset_root, camera, f"{pid:04d}")
        candidates.extend(list_images(identity_dir))

    if not candidates:
        return None

    index = min(max(image_index, 0), len(candidates) - 1)
    return candidates[index]


def collect_selected_paths(
    dataset_root: str,
) -> Tuple[List[str], List[str]]:
    visible_paths = [os.path.abspath(p) for p in MANUAL_VISIBLE_PATHS]
    infrared_paths = [os.path.abspath(p) for p in MANUAL_INFRARED_PATHS]

    if not visible_paths:
        for pid in SELECTED_IDS:
            path = find_image_for_id(dataset_root, pid, VIS_CAMS, IMAGE_INDEX)
            if path is None:
                print(f"Warning: ID {pid} 未找到 Visible 图像。")
            else:
                visible_paths.append(path)

    if not infrared_paths:
        for pid in SELECTED_IDS:
            path = find_image_for_id(dataset_root, pid, IR_CAMS, IMAGE_INDEX)
            if path is None:
                print(f"Warning: ID {pid} 未找到 Infrared 图像。")
            else:
                infrared_paths.append(path)

    for path in visible_paths + infrared_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"图像不存在：{path}")

    if not visible_paths or not infrared_paths:
        raise RuntimeError("Visible 或 Infrared 图片为空。")

    return visible_paths, infrared_paths


def load_display_image(path: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(
        image,
        (IMG_W, IMG_H),
        method=PIL_BILINEAR,
        centering=(0.5, 0.5),
    )


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return transform_test(image).unsqueeze(0).to(device)


# ============================================================
# 5. Hook 目标层
# ============================================================
def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    current: Any = model

    for token in path.split("."):
        if token.isdigit():
            current = current[int(token)]
        else:
            if not hasattr(current, token):
                names = [name for name, _ in model.named_modules()]
                close_matches = [name for name in names if token.lower() in name.lower()][:20]
                raise AttributeError(
                    f"找不到目标层：{path}\n"
                    f"失败位置：{token}\n"
                    f"可能相关的模块：{close_matches}"
                )
            current = getattr(current, token)

    if not isinstance(current, torch.nn.Module):
        raise TypeError(f"{path} 不是 torch.nn.Module。")

    return current


def find_first_4d_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj if obj.dim() == 4 else None

    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = find_first_4d_tensor(item)
            if found is not None:
                return found

    if isinstance(obj, dict):
        for item in obj.values():
            found = find_first_4d_tensor(item)
            if found is not None:
                return found

    return None


class FeatureHook:
    def __init__(self):
        self.features: List[torch.Tensor] = []

    def clear(self) -> None:
        self.features.clear()

    def __call__(self, module, inputs, output):
        feature = find_first_4d_tensor(output)
        if feature is not None:
            self.features.append(feature.detach())

    def get_last(self) -> torch.Tensor:
        if not self.features:
            raise RuntimeError(
                "Hook 没有捕获到 4D 特征。请把 TARGET_LAYER 改为输出 [B,C,H,W] 的层。"
            )
        return self.features[-1]


# ============================================================
# 6. Heatmap 生成
# ============================================================
def feature_to_response_map(feature: torch.Tensor) -> np.ndarray:
    if feature.dim() != 4:
        raise ValueError(f"目标特征必须是 4D，当前为 {tuple(feature.shape)}")

    feature = feature.float()

    # DEEN/DEE 可能把一张图扩展成多个分支，这里先做分支平均。
    if feature.size(0) > 1:
        feature = feature.mean(dim=0, keepdim=True)

    if RESPONSE_MODE == "square_mean":
        response = feature.pow(2).mean(dim=1, keepdim=True)
    elif RESPONSE_MODE == "abs_mean":
        response = feature.abs().mean(dim=1, keepdim=True)
    elif RESPONSE_MODE == "max_abs":
        response = feature.abs().max(dim=1, keepdim=True).values
    else:
        raise ValueError("RESPONSE_MODE 无效。")

    response = F.interpolate(
        response,
        size=(IMG_H, IMG_W),
        mode="bilinear",
        align_corners=False,
    )

    arr = response[0, 0].detach().cpu().numpy().astype(np.float32)
    valid = arr[np.isfinite(arr)]

    if valid.size == 0:
        raise RuntimeError("Heatmap 全部为非有限值。")

    low = float(np.percentile(valid, PERCENTILE_LOW))
    high = float(np.percentile(valid, PERCENTILE_HIGH))

    if high <= low:
        low = float(valid.min())
        high = float(valid.max())

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-12)
    arr[~np.isfinite(arr)] = 0.0

    return arr


def heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    cmap = matplotlib.colormaps.get_cmap(COLORMAP)
    rgba = cmap(np.clip(heatmap, 0.0, 1.0))
    rgb = np.uint8(np.clip(rgba[..., :3] * 255.0, 0, 255))
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(
    image: Image.Image,
    heatmap_rgb: Image.Image,
    alpha: float,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.asarray(heatmap_rgb.resize(image.size, PIL_BILINEAR), dtype=np.float32)
    overlay = (1.0 - alpha) * base + alpha * color
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)), mode="RGB")


def forward_and_make_heatmap(
    net: torch.nn.Module,
    hook: FeatureHook,
    image: Image.Image,
    modal: int,
    device: torch.device,
) -> Tuple[Image.Image, Image.Image]:
    hook.clear()
    tensor = image_to_tensor(image, device)

    with torch.no_grad():
        _ = net(tensor, tensor, modal=modal)

    feature = hook.get_last()
    response = feature_to_response_map(feature)
    heatmap_rgb = heatmap_to_rgb(response)
    overlay = overlay_heatmap(image, heatmap_rgb, OVERLAY_ALPHA)

    return heatmap_rgb, overlay


# ============================================================
# 7. 矩形遮挡
# ============================================================
def make_occluded_image(
    image: Image.Image,
    sample_index: int,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    image = image.copy().convert("RGB")
    width, height = image.size

    box_width = max(8, int(round(width * OCCLUSION_WIDTH_RATIO)))
    box_height = max(16, int(round(height * OCCLUSION_HEIGHT_RATIO)))

    if RANDOM_OCCLUSION_POSITION:
        rng = random.Random(SEED + sample_index * 9973)
        center_x = rng.uniform(0.34, 0.66) * width
        center_y = rng.uniform(0.28, 0.72) * height
    else:
        center_x = 0.5 * width
        center_y = 0.5 * height

    x1 = int(round(center_x - box_width / 2))
    y1 = int(round(center_y - box_height / 2))
    x1 = max(0, min(width - box_width, x1))
    y1 = max(0, min(height - box_height, y1))
    x2 = x1 + box_width
    y2 = y1 + box_height

    ImageDraw.Draw(image).rectangle([x1, y1, x2, y2], fill=OCCLUSION_COLOR)
    return image, (x1, y1, x2, y2)


# ============================================================
# 8. 单张样本处理与保存
# ============================================================
def safe_stem(path: str) -> str:
    parts = Path(path).parts[-3:]
    return "_".join(parts).replace(".", "_")


def process_one_image(
    net: torch.nn.Module,
    hook: FeatureHook,
    image_path: str,
    modal: int,
    modality_name: str,
    sample_index: int,
    device: torch.device,
    output_dir: str,
) -> Dict[str, Any]:
    original = load_display_image(image_path)

    original_heatmap, original_overlay = forward_and_make_heatmap(
        net, hook, original, modal, device
    )

    occluded, box = make_occluded_image(original, sample_index)

    occluded_heatmap, occluded_overlay = forward_and_make_heatmap(
        net, hook, occluded, modal, device
    )

    stem = safe_stem(image_path)
    sample_dir = os.path.join(output_dir, modality_name)
    os.makedirs(sample_dir, exist_ok=True)

    original.save(os.path.join(sample_dir, f"{stem}_original.png"))
    original_heatmap.save(os.path.join(sample_dir, f"{stem}_heatmap.png"))
    original_overlay.save(os.path.join(sample_dir, f"{stem}_overlay.png"))
    occluded.save(os.path.join(sample_dir, f"{stem}_occluded.png"))
    occluded_heatmap.save(os.path.join(sample_dir, f"{stem}_occluded_heatmap.png"))
    occluded_overlay.save(os.path.join(sample_dir, f"{stem}_occluded_overlay.png"))

    print(f"[{modality_name}] {image_path}")
    print(f"  occlusion box={box}")

    return {
        "path": image_path,
        "original": original,
        "heatmap": original_heatmap,
        "overlay": original_overlay,
        "occluded": occluded,
        "occluded_heatmap": occluded_heatmap,
        "occluded_overlay": occluded_overlay,
    }


# ============================================================
# 9. 论文风格总图
# ============================================================
def hide_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_grid(
    visible_results: Sequence[Dict[str, Any]],
    infrared_results: Sequence[Dict[str, Any]],
    output_dir: str,
) -> Tuple[str, str]:
    all_results = list(visible_results) + list(infrared_results)
    num_vis = len(visible_results)
    num_ir = len(infrared_results)
    num_cols = len(all_results)
    num_rows = 4

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(CELL_WIDTH * num_cols + 1.15, CELL_HEIGHT * num_rows + 0.70),
        squeeze=False,
        gridspec_kw={"wspace": WSPACE, "hspace": HSPACE},
    )

    row_keys = ["original", "heatmap", "occluded", "occluded_heatmap"]
    row_labels = ["(a)", "(b)", "(c)", "(d)"]

    for col, item in enumerate(all_results):
        for row, key in enumerate(row_keys):
            axes[row, col].imshow(item[key], aspect="auto")
            hide_axes(axes[row, col])

    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.34, 0.50, label,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            transform=axes[row, 0].transAxes,
            clip_on=False,
        )

    plt.subplots_adjust(left=0.085, right=0.995, top=0.935, bottom=0.025)

    if num_vis > 0:
        left = axes[0, 0].get_position().x0
        right = axes[0, num_vis - 1].get_position().x1
        fig.text((left + right) / 2, 0.965, "Visible Images",
                 ha="center", va="center", fontsize=13, fontweight="bold")

    if num_ir > 0:
        left = axes[0, num_vis].get_position().x0
        right = axes[0, -1].get_position().x1
        fig.text((left + right) / 2, 0.965, "Infrared Images",
                 ha="center", va="center", fontsize=13, fontweight="bold")

    if num_vis > 0 and num_ir > 0:
        left_box = axes[0, num_vis - 1].get_position()
        right_box = axes[0, num_vis].get_position()
        separator_x = (left_box.x1 + right_box.x0) / 2

        fig.add_artist(Line2D(
            [separator_x, separator_x],
            [axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1],
            transform=fig.transFigure,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
        ))

    pdf_path = os.path.join(output_dir, OUTPUT_BASENAME + ".pdf")
    png_path = os.path.join(output_dir, OUTPUT_BASENAME + ".png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Saved grid PDF:", pdf_path)
    print("Saved grid PNG:", png_path)
    return pdf_path, png_path


# ============================================================
# 10. 主流程
# ============================================================
def main() -> None:
    set_seed(SEED)

    device = resolve_device(args.gpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        cudnn.benchmark = True

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"SYSU 数据集路径不存在：{dataset_root}")

    output_dir = os.path.join(
        os.path.abspath(args.output_root),
        MODEL_DISPLAY_NAME,
        args.target_layer.replace(".", "_"),
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 78)
    print("Model name   :", MODEL_DISPLAY_NAME)
    print("Device       :", device)
    print("SYSU path    :", dataset_root)
    print("Target layer :", args.target_layer)
    print("Output dir   :", output_dir)
    print("=" * 78)

    net = build_model(device)
    target_module = get_module_by_path(net, args.target_layer)

    hook = FeatureHook()
    hook_handle = target_module.register_forward_hook(hook)

    visible_paths, infrared_paths = collect_selected_paths(dataset_root)

    visible_results = []
    infrared_results = []

    try:
        for index, image_path in enumerate(visible_paths):
            visible_results.append(process_one_image(
                net, hook, image_path, 1, "visible", index, device, output_dir
            ))

        offset = len(visible_paths)
        for index, image_path in enumerate(infrared_paths):
            infrared_results.append(process_one_image(
                net, hook, image_path, 2, "infrared",
                offset + index, device, output_dir
            ))
    finally:
        hook_handle.remove()

    draw_grid(visible_results, infrared_results, output_dir)
    print("Heatmap visualization completed successfully.")


if __name__ == "__main__":
    main()
