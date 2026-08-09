#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYSU-MM01 检索结果可视化：DEEN vs MDGRN

使用方法
--------
1. 先把 ACTIVE_MODEL = "deen"，运行一次，保存 DEEN 检索结果；
2. 再把 ACTIVE_MODEL = "ours"，运行一次，保存 MDGRN 检索结果；
3. 当两个缓存文件都存在时，脚本会自动生成上下两行的 Top-10 对比图。

正常情况下，之后切换模型时只需要改一行：
    ACTIVE_MODEL = "deen"
或：
    ACTIVE_MODEL = "ours"

说明
----
- 查询集：SYSU-MM01 测试身份的全部红外图像（cam3、cam6）。
- 图库集：All-search single-shot；每个测试身份在每个可见光相机
  （cam1、cam2、cam4、cam5）随机抽取 1 张图像。
- 绿色边框：与查询图像身份相同。
- 红色边框：与查询图像身份不同。
- 两次运行使用相同的 TRIAL 和 SEED，因此 query/gallery 完全一致。
"""

import os
import sys
import random
import importlib.util
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ============================================================
# 1. 只需切换这一项："deen" 或 "ours"
# ============================================================
ACTIVE_MODEL = "ours"


# ============================================================
# 2. 路径与模型配置
#    两个工程路径只需在第一次使用时确认一次。
# ============================================================
SYSU_PATH = "./Datasets/SYSU-MM01/"

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "deen": {
        # 原始 DEEN 工程中必须存在 model.py
        "project_root": "/DEEN",
        "checkpoint": (
            "/"
        ),
        "display_name": "DEEN",
    },
    "ours": {
        # Update this path if the improved model project is stored elsewhere.
        "project_root": "/MDGRN",
        "checkpoint": (
            ""
        ),
        "display_name": "MDGRN",
    },
}


# ============================================================
# 3. 测试与绘图配置
# ============================================================
DEVICE = "cuda:0"                 # 例如 cuda:0、cuda:3 或 cpu
N_CLASS = 395
ARCH = "resnet50"

IMG_H = 384
IMG_W = 144
BATCH_SIZE = 64
NUM_WORKERS = 4
FEATURE_ALPHA = 0.3               # pool 与 fc 的融合比例

MODE = "all"                      # 当前脚本按 SYSU All-search 构建图库
TRIAL = 0
SEED = 0
TOPK = 10
NUM_QUERY_EXAMPLES = 2

MANUAL_QUERY_INDICES = None
MANUAL_QUERY_IDS = [83,10]

CACHE_DIR = "save_retrieval"
OUTPUT_DIR = "save_retrieval"
OUTPUT_BASENAME = "sysu_deen_vs_mdgrn_top10"

VIS_CAMS = ("cam1", "cam2", "cam4", "cam5")
IR_CAMS = ("cam3", "cam6")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ============================================================
# 4. 基础工具
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA 不可用，自动改用 CPU。")
        return torch.device("cpu")
    return torch.device(device_name)


def list_images(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []

    paths = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(paths)


def read_test_ids(sysu_path: str) -> List[int]:
    test_id_path = os.path.join(sysu_path, "exp", "test_id.txt")
    if not os.path.isfile(test_id_path):
        raise FileNotFoundError(
            f"未找到 SYSU 测试身份文件：{test_id_path}\n"
            "请确认 SYSU_PATH 指向 SYSU-MM01 数据集根目录。"
        )

    with open(test_id_path, "r", encoding="utf-8") as file:
        text = file.read().replace("\n", ",")

    ids = []
    for token in text.split(","):
        token = token.strip()
        if token:
            ids.append(int(token))

    if not ids:
        raise RuntimeError(f"测试身份文件为空：{test_id_path}")

    return sorted(set(ids))


# ============================================================
# 5. 构建 SYSU 标准 Query / Gallery
# ============================================================
def build_sysu_query(
    sysu_path: str,
    test_ids: Sequence[int],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """All-search 查询集：测试身份在 cam3、cam6 中的全部红外图像。"""
    query_paths: List[str] = []
    query_labels: List[int] = []
    query_cams: List[int] = []

    for pid in test_ids:
        for camera in IR_CAMS:
            camera_id = int(camera.replace("cam", ""))
            identity_dir = os.path.join(sysu_path, camera, f"{pid:04d}")

            for image_path in list_images(identity_dir):
                query_paths.append(image_path)
                query_labels.append(pid)
                query_cams.append(camera_id)

    if not query_paths:
        raise RuntimeError("没有找到查询图像，请检查 SYSU_PATH 和目录结构。")

    return (
        query_paths,
        np.asarray(query_labels, dtype=np.int64),
        np.asarray(query_cams, dtype=np.int64),
    )


def build_sysu_gallery(
    sysu_path: str,
    test_ids: Sequence[int],
    seed: int,
    trial: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    All-search single-shot 图库：
    每个测试身份在 cam1、cam2、cam4、cam5 中各随机抽取 1 张。
    """
    rng = random.Random(seed + trial)

    gallery_paths: List[str] = []
    gallery_labels: List[int] = []
    gallery_cams: List[int] = []

    for pid in test_ids:
        for camera in VIS_CAMS:
            camera_id = int(camera.replace("cam", ""))
            identity_dir = os.path.join(sysu_path, camera, f"{pid:04d}")
            candidates = list_images(identity_dir)

            if not candidates:
                continue

            selected_path = candidates[rng.randrange(len(candidates))]
            gallery_paths.append(selected_path)
            gallery_labels.append(pid)
            gallery_cams.append(camera_id)

    if not gallery_paths:
        raise RuntimeError("没有找到图库图像，请检查 SYSU_PATH 和目录结构。")

    return (
        gallery_paths,
        np.asarray(gallery_labels, dtype=np.int64),
        np.asarray(gallery_cams, dtype=np.int64),
    )


# ============================================================
# 6. 动态加载对应工程中的 model.py
# ============================================================
def import_embed_net(project_root: str, model_key: str):
    project_root = os.path.abspath(project_root)
    model_file = os.path.join(project_root, "model.py")

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"未找到模型文件：{model_file}\n"
            f"请检查 MODEL_CONFIGS['{model_key}']['project_root']。"
        )

    # 让 model.py 中的本地依赖（如 resnet.py）可以被正常导入。
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module_name = f"sysu_retrieval_model_{model_key}"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载：{model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "embed_net"):
        raise AttributeError(f"{model_file} 中没有找到 embed_net。")

    return module.embed_net


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_model(
    model_key: str,
    device: torch.device,
):
    if model_key not in MODEL_CONFIGS:
        raise KeyError(
            f"ACTIVE_MODEL={model_key!r} 无效，只能使用：{list(MODEL_CONFIGS)}"
        )

    config = MODEL_CONFIGS[model_key]
    project_root = os.path.abspath(config["project_root"])
    checkpoint_path = config["checkpoint"]

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"未找到权重：{checkpoint_path}\n"
            f"请检查 MODEL_CONFIGS['{model_key}']['checkpoint']。"
        )

    embed_net = import_embed_net(project_root, model_key)

    try:
        net = embed_net(N_CLASS, dataset="sysu", arch=ARCH)
    except TypeError:
        # 兼容少数旧代码的构造函数写法。
        net = embed_net(N_CLASS, "sysu", arch=ARCH)

    print(f"Loading {config['display_name']} model:")
    print(f"  project_root: {project_root}")
    print(f"  checkpoint  : {checkpoint_path}")

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
        raise TypeError("checkpoint 中没有可用的 state_dict。")

    state_dict = strip_module_prefix(state_dict)

    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "权重与当前 project_root 下的 model.py 结构不一致。\n"
            "不要使用 strict=False 强行加载，否则新增模块会保持随机参数。\n"
            f"请把 {model_key} 的 project_root 指向训练该权重时使用的原工程。\n\n"
            f"原始错误：\n{error}"
        ) from error

    net = net.to(device)
    net.eval()

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("  checkpoint epoch:", checkpoint["epoch"])

    return net, config


# ============================================================
# 7. 图像数据与批量特征提取
# ============================================================
transform_test = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str]):
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"读取图像失败：{image_path}") from error

        return transform_test(image)


def prepare_feature_tensor(feature: torch.Tensor) -> torch.Tensor:
    """把模型输出统一为 [B, D] 并逐样本 L2 归一化。"""
    if not torch.is_tensor(feature):
        raise TypeError(f"模型特征必须是 Tensor，当前为：{type(feature)}")

    if feature.dim() == 1:
        feature = feature.unsqueeze(0)
    elif feature.dim() == 4:
        feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
    elif feature.dim() > 2:
        feature = feature.flatten(1)

    if feature.dim() != 2:
        raise ValueError(f"无法处理模型输出形状：{tuple(feature.shape)}")

    return F.normalize(feature, p=2, dim=1, eps=1e-12)


def merge_model_outputs(outputs) -> torch.Tensor:
    """
    兼容：
    - 单个 Tensor；
    - (feat_pool, feat_fc)；
    - 返回更多内容但前两个是测试特征的情况。
    """
    if torch.is_tensor(outputs):
        return prepare_feature_tensor(outputs)

    if not isinstance(outputs, (tuple, list)) or len(outputs) == 0:
        raise ValueError("无法识别模型测试阶段的输出格式。")

    feat_pool = prepare_feature_tensor(outputs[0])

    if len(outputs) == 1:
        return feat_pool

    feat_fc = prepare_feature_tensor(outputs[1])

    if feat_pool.shape != feat_fc.shape:
        raise ValueError(
            "模型前两个输出维度不一致，无法按 FEATURE_ALPHA 融合："
            f"{tuple(feat_pool.shape)} vs {tuple(feat_fc.shape)}"
        )

    feature = (
        FEATURE_ALPHA * feat_pool
        + (1.0 - FEATURE_ALPHA) * feat_fc
    )
    return F.normalize(feature, p=2, dim=1, eps=1e-12)


def extract_features(
    net,
    image_paths,
    modal,
    device,
    description,
):
    dataset = ImagePathDataset(image_paths)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    feature_batches = []
    total_batches = len(loader)

    with torch.no_grad():
        for batch_index, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)

            input_batch_size = images.size(0)

            # DEEN 测试前向
            outputs = net(images, images, modal=modal)

            # outputs[0] 和 outputs[1] 通常都是 [3B, 2048]
            features = merge_model_outputs(outputs)

            output_batch_size = features.size(0)

            # =====================================================
            # DEEN 的 DEE 会将每张图片扩展为 3 个分支：
            # [branch1 的 B 张, branch2 的 B 张, branch3 的 B 张]
            # 将三个分支融合回 [B, 2048]
            # =====================================================
            if output_batch_size == input_batch_size * 3:
                branch1, branch2, branch3 = torch.chunk(
                    features,
                    chunks=3,
                    dim=0
                )

                # 三分支平均融合
                features = (
                    branch1
                    + branch2
                    + branch3
                ) / 3.0

                # 融合后重新归一化
                features = F.normalize(
                    features,
                    p=2,
                    dim=1,
                    eps=1e-12
                )

            elif output_batch_size == input_batch_size:
                # 普通 ResNet50 或已经返回单分支特征的模型
                pass

            else:
                raise RuntimeError(
                    f"{description} 模型输出数量异常："
                    f"输入 batch={input_batch_size}，"
                    f"输出 feature={output_batch_size}，"
                    f"feature shape={tuple(features.shape)}"
                )

            feature_batches.append(
                features.detach().cpu().numpy().astype(np.float32)
            )

            if (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == total_batches
            ):
                print(
                    f"{description}: "
                    f"batch {batch_index:03d}/{total_batches:03d}, "
                    f"input={input_batch_size}, "
                    f"feature={tuple(features.shape)}"
                )

    if not feature_batches:
        raise RuntimeError(
            f"{description} 没有提取到任何特征。"
        )

    all_features = np.concatenate(
        feature_batches,
        axis=0
    )

    if all_features.shape[0] != len(image_paths):
        raise RuntimeError(
            f"{description} 数量不一致："
            f"特征 {all_features.shape[0]}，"
            f"图像 {len(image_paths)}"
        )

    print(
        f"{description} extraction finished: "
        f"{all_features.shape}"
    )

    return all_features

# ============================================================
# 8. 保存、读取当前模型的检索结果
# ============================================================
def cache_path(model_key: str) -> str:
    return os.path.join(CACHE_DIR, f"sysu_{model_key}_trial{TRIAL}.npz")


def save_retrieval_cache(
    model_key: str,
    display_name: str,
    similarity: np.ndarray,
    query_paths: Sequence[str],
    query_labels: np.ndarray,
    query_cams: np.ndarray,
    gallery_paths: Sequence[str],
    gallery_labels: np.ndarray,
    gallery_cams: np.ndarray,
) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    save_path = cache_path(model_key)

    np.savez_compressed(
        save_path,
        model_key=np.asarray(model_key),
        display_name=np.asarray(display_name),
        similarity=np.asarray(similarity, dtype=np.float32),
        query_paths=np.asarray(query_paths, dtype=str),
        query_labels=np.asarray(query_labels, dtype=np.int64),
        query_cams=np.asarray(query_cams, dtype=np.int64),
        gallery_paths=np.asarray(gallery_paths, dtype=str),
        gallery_labels=np.asarray(gallery_labels, dtype=np.int64),
        gallery_cams=np.asarray(gallery_cams, dtype=np.int64),
        seed=np.asarray(SEED),
        trial=np.asarray(TRIAL),
    )

    print("Saved retrieval cache:", save_path)
    print("  similarity:", similarity.shape)
    return save_path


def load_cache(model_key: str):
    path = cache_path(model_key)
    if not os.path.isfile(path):
        return None
    return np.load(path, allow_pickle=False)


# ============================================================
# 9. 查询样本选择
# ============================================================
def first_correct_rank(
    ranked_indices: np.ndarray,
    query_label: int,
    gallery_labels: np.ndarray,
) -> int:
    positions = np.flatnonzero(gallery_labels[ranked_indices] == query_label)
    if positions.size == 0:
        return len(ranked_indices) + 1
    return int(positions[0]) + 1


def query_improvement_score(
    query_index: int,
    deen_rank: np.ndarray,
    ours_rank: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    topk: int,
):
    query_label = int(query_labels[query_index])
    deen_top = deen_rank[query_index, :topk]
    ours_top = ours_rank[query_index, :topk]

    deen_correct = int(np.sum(gallery_labels[deen_top] == query_label))
    ours_correct = int(np.sum(gallery_labels[ours_top] == query_label))

    deen_first = first_correct_rank(deen_top, query_label, gallery_labels)
    ours_first = first_correct_rank(ours_top, query_label, gallery_labels)

    correct_gain = ours_correct - deen_correct
    rank_gain = deen_first - ours_first

    # Top-10 正确数量的提升优先；首个正确结果提前作为次级指标。
    score = 100.0 * correct_gain + 5.0 * rank_gain + ours_correct

    return {
        "query_index": query_index,
        "query_label": query_label,
        "deen_correct": deen_correct,
        "ours_correct": ours_correct,
        "deen_first": deen_first,
        "ours_first": ours_first,
        "score": score,
    }


def select_query_indices(
    deen_rank: np.ndarray,
    ours_rank: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> List[int]:
    num_queries = len(query_labels)

    if MANUAL_QUERY_INDICES is not None:
        selected = [int(index) for index in MANUAL_QUERY_INDICES]
        for index in selected:
            if index < 0 or index >= num_queries:
                raise IndexError(f"手动查询索引越界：{index}")
        return selected[:NUM_QUERY_EXAMPLES]

    all_items = [
        query_improvement_score(
            query_index=index,
            deen_rank=deen_rank,
            ours_rank=ours_rank,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            topk=TOPK,
        )
        for index in range(num_queries)
    ]

    if MANUAL_QUERY_IDS is not None:
        selected = []
        for pid in MANUAL_QUERY_IDS:
            candidates = [item for item in all_items if item["query_label"] == int(pid)]
            if not candidates:
                print(f"Warning: 查询集中没有找到 ID {pid}。")
                continue
            candidates.sort(key=lambda item: item["score"], reverse=True)
            selected.append(int(candidates[0]["query_index"]))
        if selected:
            return selected[:NUM_QUERY_EXAMPLES]

    all_items.sort(key=lambda item: item["score"], reverse=True)

    selected: List[int] = []
    used_ids = set()

    for item in all_items:
        improved = (
            item["ours_correct"] > item["deen_correct"]
            or item["ours_first"] < item["deen_first"]
        )
        if not improved:
            continue
        if item["query_label"] in used_ids:
            continue

        selected.append(int(item["query_index"]))
        used_ids.add(item["query_label"])

        if len(selected) >= NUM_QUERY_EXAMPLES:
            break

    if len(selected) < NUM_QUERY_EXAMPLES:
        for item in all_items:
            if item["query_index"] in selected:
                continue
            if item["query_label"] in used_ids:
                continue

            selected.append(int(item["query_index"]))
            used_ids.add(item["query_label"])

            if len(selected) >= NUM_QUERY_EXAMPLES:
                break

    print("Selected queries:")
    for index in selected:
        item = query_improvement_score(
            index,
            deen_rank,
            ours_rank,
            query_labels,
            gallery_labels,
            TOPK,
        )
        print(
            f"  index={item['query_index']}, ID={item['query_label']}, "
            f"DEEN correct={item['deen_correct']}, "
            f"MDGRN correct={item['ours_correct']}, "
            f"DEEN first=R-{item['deen_first']}, "
            f"MDGRN first=R-{item['ours_first']}"
        )

    return selected


# ============================================================
# 10. 绘制紧凑、等尺寸的论文风格 Top-10 对比图
# ============================================================
# 每个单元格的物理尺寸。需要更紧凑时只调这几项即可。
FIG_CELL_W = 0.68
FIG_CELL_H = 1.56
FIG_WSPACE = 0.028
FIG_HSPACE = 0.030

# Pillow 新旧版本兼容。
_RESAMPLE_BILINEAR = (
    Image.Resampling.BILINEAR
    if hasattr(Image, "Resampling")
    else Image.BILINEAR
)


def show_image(ax, image_path: str) -> None:
    """
    将所有 Query / Gallery 图像统一成相同大小后铺满 Axes。

    ImageOps.fit 会保持人体比例，并在必要时从边缘轻微裁剪；
    aspect='auto' 可防止 Matplotlib 根据原图宽高比缩小某些子图。
    """
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        image = ImageOps.fit(
            image,
            (IMG_W, IMG_H),
            method=_RESAMPLE_BILINEAR,
            centering=(0.5, 0.5),
        )

    ax.imshow(
        image,
        aspect="auto",
        interpolation="bilinear",
    )
    ax.set_aspect("auto")
    ax.set_xlim(-0.5, IMG_W - 0.5)
    ax.set_ylim(IMG_H - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)


def set_border(ax, color: str, linewidth: float = 2.0, linestyle: str = "-") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)
        spine.set_linestyle(linestyle)


def draw_comparison_figure(
    query_paths: np.ndarray,
    query_labels: np.ndarray,
    gallery_paths: np.ndarray,
    gallery_labels: np.ndarray,
    deen_rank: np.ndarray,
    ours_rank: np.ndarray,
    selected_query_indices: Sequence[int],
) -> Tuple[str, str]:
    """
    行顺序按模型分组，而不是按 Query 交替：

        DEEN  - Query 1
        DEEN  - Query 2
        ...
        MDGRN - Query 1
        MDGRN - Query 2
        ...

    内部模型键仍保留为 ``ours``，因此旧缓存
    ``sysu_ours_trial0.npz`` 无需重新生成。
    """
    model_groups = [
        (MODEL_CONFIGS["deen"]["display_name"], deen_rank),
        (MODEL_CONFIGS["ours"]["display_name"], ours_rank),
    ]

    num_examples = len(selected_query_indices)
    num_models = len(model_groups)
    num_rows = num_examples * num_models
    num_cols = TOPK + 1

    # 所有列完全等宽，Query 与 Top-10 尺寸一致。
    fig_width = FIG_CELL_W * num_cols + 0.78
    fig_height = FIG_CELL_H * num_rows + 0.62

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={
            "width_ratios": [1.0] * num_cols,
            "wspace": FIG_WSPACE,
            "hspace": FIG_HSPACE,
        },
    )

    green = "#00A651"
    red = "#F21D2F"
    query_border = "#777777"

    # 先画完 DEEN 的所有 Query，再画 MDGRN 的所有 Query。
    for model_index, (_, rank_matrix) in enumerate(model_groups):
        for example_order, query_index in enumerate(selected_query_indices):
            row = model_index * num_examples + example_order
            query_label = int(query_labels[query_index])

            query_ax = axes[row, 0]
            show_image(query_ax, str(query_paths[query_index]))
            set_border(
                query_ax,
                query_border,
                linewidth=1.05,
                linestyle="--",
            )

            top_indices = rank_matrix[query_index, :TOPK]

            for rank_position, gallery_index in enumerate(top_indices, start=1):
                ax = axes[row, rank_position]
                gallery_index = int(gallery_index)
                show_image(ax, str(gallery_paths[gallery_index]))

                correct = int(gallery_labels[gallery_index]) == query_label
                set_border(
                    ax,
                    green if correct else red,
                    linewidth=1.75,
                )

    # 左侧为两个模型组各留一个统一标签，不再每行重复写模型名。
    plt.subplots_adjust(
        left=0.082,
        right=0.995,
        top=0.910,
        bottom=0.025,
    )

    first_query_x = axes[0, 0].get_position().x0
    label_x = max(first_query_x - 0.037, 0.012)

    for model_index, (model_name, _) in enumerate(model_groups):
        first_row = model_index * num_examples
        last_row = first_row + num_examples - 1
        group_top = axes[first_row, 0].get_position().y1
        group_bottom = axes[last_row, 0].get_position().y0
        group_center = (group_top + group_bottom) / 2.0

        fig.text(
            label_x,
            group_center,
            model_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.8,
            fontweight="bold",
        )

    # Query 与检索结果之间的竖虚线。
    query_box = axes[0, 0].get_position()
    result_box = axes[0, 1].get_position()
    separator_x = (query_box.x1 + result_box.x0) / 2.0

    fig.add_artist(Line2D(
        [separator_x, separator_x],
        [axes[-1, 0].get_position().y0, axes[0, 0].get_position().y1],
        transform=fig.transFigure,
        color="#777777",
        linestyle="--",
        linewidth=0.9,
    ))

    # 只在 DEEN 与 MDGRN 两个模型块之间画横虚线。
    for model_index in range(num_models - 1):
        upper_last_row = (model_index + 1) * num_examples - 1
        lower_first_row = upper_last_row + 1
        upper_bottom = axes[upper_last_row, 0].get_position().y0
        lower_top = axes[lower_first_row, 0].get_position().y1
        separator_y = (upper_bottom + lower_top) / 2.0

        fig.add_artist(Line2D(
            [axes[0, 0].get_position().x0, axes[0, -1].get_position().x1],
            [separator_y, separator_y],
            transform=fig.transFigure,
            color="#777777",
            linestyle="--",
            linewidth=0.95,
        ))

    # 顶部标题与箭头。
    first_result_box = axes[0, 1].get_position()
    last_result_box = axes[0, -1].get_position()
    top_image_y = axes[0, 0].get_position().y1
    arrow_y = min(top_image_y + 0.018, 0.965)

    axes[0, 0].annotate(
        "",
        xy=(last_result_box.x1, arrow_y),
        xytext=(first_result_box.x0, arrow_y),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.15,
            "color": "black",
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )

    fig.text(
        (first_result_box.x0 + last_result_box.x1) / 2.0,
        arrow_y + 0.010,
        f"Retrieval results from R-1 to R-{TOPK}",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME + ".pdf")
    png_path = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME + ".png")

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)

    print("Saved comparison PDF:", pdf_path)
    print("Saved comparison PNG:", png_path)
    return pdf_path, png_path


# ============================================================
# 11. 检查两个缓存并自动绘图
# ============================================================
def compare_and_draw_if_ready() -> bool:
    deen_data = load_cache("deen")
    ours_data = load_cache("ours")

    if deen_data is None or ours_data is None:
        missing = []
        if deen_data is None:
            missing.append("deen")
        if ours_data is None:
            missing.append("ours")

        print("\n当前还缺少缓存：", ", ".join(missing))
        print("请切换 ACTIVE_MODEL 后再运行一次。")
        return False

    fields_to_check = [
        "query_paths",
        "query_labels",
        "query_cams",
        "gallery_paths",
        "gallery_labels",
        "gallery_cams",
    ]

    for field in fields_to_check:
        if not np.array_equal(deen_data[field], ours_data[field]):
            raise RuntimeError(
                f"DEEN 与 MDGRN 的 {field} 不一致，不能直接比较。\n"
                "请确保两次运行的 SYSU_PATH、SEED、TRIAL、MODE 完全一致，"
                "然后删除 save_retrieval 中旧的 npz 文件重新运行。"
            )

    deen_similarity = np.asarray(deen_data["similarity"], dtype=np.float32)
    ours_similarity = np.asarray(ours_data["similarity"], dtype=np.float32)

    if deen_similarity.shape != ours_similarity.shape:
        raise RuntimeError(
            f"两个相似度矩阵形状不同："
            f"DEEN={deen_similarity.shape}, MDGRN={ours_similarity.shape}"
        )

    deen_rank = np.argsort(-deen_similarity, axis=1)
    ours_rank = np.argsort(-ours_similarity, axis=1)

    selected_query_indices = select_query_indices(
        deen_rank=deen_rank,
        ours_rank=ours_rank,
        query_labels=ours_data["query_labels"],
        gallery_labels=ours_data["gallery_labels"],
    )

    if not selected_query_indices:
        raise RuntimeError("没有选出可视化查询图像。")

    draw_comparison_figure(
        query_paths=ours_data["query_paths"],
        query_labels=ours_data["query_labels"],
        gallery_paths=ours_data["gallery_paths"],
        gallery_labels=ours_data["gallery_labels"],
        deen_rank=deen_rank,
        ours_rank=ours_rank,
        selected_query_indices=selected_query_indices,
    )
    return True


# ============================================================
# 12. 主流程
# ============================================================
def main() -> None:
    if MODE.lower() != "all":
        raise ValueError("当前最终版脚本仅实现 SYSU All-search，请保持 MODE='all'。")

    set_seed(SEED)
    device = resolve_device(DEVICE)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        cudnn.benchmark = True

    print("=" * 70)
    print("ACTIVE_MODEL:", ACTIVE_MODEL)
    print("DEVICE      :", device)
    print("SYSU_PATH   :", SYSU_PATH)
    print("TRIAL/SEED  :", TRIAL, SEED)
    print("=" * 70)

    test_ids = read_test_ids(SYSU_PATH)
    print("Number of test IDs:", len(test_ids))

    query_paths, query_labels, query_cams = build_sysu_query(
        SYSU_PATH,
        test_ids,
    )
    gallery_paths, gallery_labels, gallery_cams = build_sysu_gallery(
        SYSU_PATH,
        test_ids,
        seed=SEED,
        trial=TRIAL,
    )

    print("Query images  :", len(query_paths))
    print("Gallery images:", len(gallery_paths))

    net, config = load_model(ACTIVE_MODEL, device)

    # SYSU 常用测试设置：IR query 使用 modal=2，VIS gallery 使用 modal=1。
    print("\nExtracting infrared query features...")
    query_features = extract_features(
        net=net,
        image_paths=query_paths,
        modal=2,
        device=device,
        description="Query",
    )

    print("\nExtracting visible gallery features...")
    gallery_features = extract_features(
        net=net,
        image_paths=gallery_paths,
        modal=1,
        device=device,
        description="Gallery",
    )

    print("Query feature shape  :", query_features.shape)
    print("Gallery feature shape:", gallery_features.shape)

    # 特征已经逐样本 L2 归一化，矩阵乘法即余弦相似度。
    similarity = np.matmul(query_features, gallery_features.T)
    similarity = np.clip(similarity, -1.0, 1.0)

    save_retrieval_cache(
        model_key=ACTIVE_MODEL,
        display_name=config["display_name"],
        similarity=similarity,
        query_paths=query_paths,
        query_labels=query_labels,
        query_cams=query_cams,
        gallery_paths=gallery_paths,
        gallery_labels=gallery_labels,
        gallery_cams=gallery_cams,
    )

    print()
    compare_and_draw_if_ready()


if __name__ == "__main__":
    main()
