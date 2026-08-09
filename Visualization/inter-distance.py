import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from model import embed_net

# ============================================================
# 1. 配置区
# ============================================================
MODEL_PATH = ""
SYSU_PATH = "./Datasets/SYSU-MM01/"

SELECTED_IDS = [
    89, 90, 93, 102, 105, 108, 112, 116, 117, 122,
    125, 129, 130, 134, 138, 139, 150, 152, 162, 166
]

K = 10
VIS_CAMS = ["cam1", "cam2", "cam4", "cam5"]
IR_CAMS = ["cam3", "cam6"]

IMG_W = 144
IMG_H = 384
DEVICE = "cuda:3"
SEED = 0
FEATURE_ALPHA = 0.5
NUM_BINS = 150
OUTPUT_DIR = "save_distance"
OUTPUT_NAME = "deen_style_distance"

# ============================================================
# 2. 随机种子
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ============================================================
# 4. 加载模型
# ============================================================
N_CLASS = 395
net = embed_net(N_CLASS, dataset="sysu", arch="resnet50").to(device)
cudnn.benchmark = True

print("Loading model:", MODEL_PATH)
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)
state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
net.load_state_dict(state_dict, strict=True)
net.eval()

if isinstance(checkpoint, dict) and "epoch" in checkpoint:
    print("Loaded checkpoint epoch:", checkpoint["epoch"])

# ============================================================
# 5. 图像读取工具
# ============================================================
def load_images_for_id(pid, cameras, base_path):
    image_lists = {}
    for camera in cameras:
        camera_dir = os.path.join(base_path, camera, f"{pid:04d}")
        if not os.path.isdir(camera_dir):
            continue
        images = sorted([
            os.path.join(camera_dir, filename)
            for filename in os.listdir(camera_dir)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        if images:
            image_lists[camera] = images
    return image_lists


def pick_evenly_from_cams(image_lists, num_images):
    cameras = sorted(image_lists.keys())
    if not cameras:
        return []

    selected = []
    image_index = 0
    while len(selected) < num_images:
        added = False
        for camera in cameras:
            camera_images = image_lists[camera]
            if image_index < len(camera_images):
                selected.append(camera_images[image_index])
                added = True
                if len(selected) >= num_images:
                    break
        if not added:
            break
        image_index += 1
    return selected

# ============================================================
# 6. 模型特征处理
# ============================================================
def collapse_model_feature(feature):
    """把 [D]、[1,D]、[K,D] 或 [K,C,H,W] 统一成一条一维特征。"""
    if not torch.is_tensor(feature):
        raise TypeError(f"模型输出必须是 Tensor，当前类型为 {type(feature)}")

    if feature.dim() == 1:
        feature = feature.unsqueeze(0)
    elif feature.dim() == 4:
        feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
    elif feature.dim() > 2:
        feature = feature.flatten(1)

    if feature.dim() != 2:
        raise ValueError(f"无法处理模型输出形状：{tuple(feature.shape)}")

    feature = F.normalize(feature, p=2, dim=1, eps=1e-12)
    feature = feature.mean(dim=0)
    feature = F.normalize(feature, p=2, dim=0, eps=1e-12)
    return feature


def extract_embedding(image_path, modal):
    """modal=1: Visible；modal=2: Infrared。"""
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image)
    image = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = net(image, image, modal=modal)

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise ValueError("模型测试阶段应返回至少两个特征输出。")

    feat_pool = collapse_model_feature(outputs[0])
    feat_fc = collapse_model_feature(outputs[1])

    if feat_pool.numel() != feat_fc.numel():
        raise ValueError(
            f"两个待融合特征维度不同：{feat_pool.shape} 与 {feat_fc.shape}"
        )

    final_feature = (
        FEATURE_ALPHA * feat_pool
        + (1.0 - FEATURE_ALPHA) * feat_fc
    )
    final_feature = F.normalize(final_feature, p=2, dim=0, eps=1e-12)
    return final_feature.cpu().numpy().astype(np.float32)

# ============================================================
# 7. 提取 VIS / IR 特征
# ============================================================
vis_features, vis_labels = [], []
ir_features, ir_labels = [], []

print("Extracting features...")
for index, pid in enumerate(SELECTED_IDS, start=1):
    print(f"[{index:02d}/{len(SELECTED_IDS):02d}] Processing ID {pid}")

    vis_images = pick_evenly_from_cams(
        load_images_for_id(pid, VIS_CAMS, SYSU_PATH), K
    )
    ir_images = pick_evenly_from_cams(
        load_images_for_id(pid, IR_CAMS, SYSU_PATH), K
    )

    if len(vis_images) == 0 or len(ir_images) == 0:
        print(f"  Warning: ID {pid} 缺少某一模态图片，跳过。")
        continue

    if len(vis_images) < K:
        print(f"  Warning: ID {pid} Visible 仅找到 {len(vis_images)} 张。")
    if len(ir_images) < K:
        print(f"  Warning: ID {pid} Infrared 仅找到 {len(ir_images)} 张。")

    for image_path in vis_images:
        vis_features.append(extract_embedding(image_path, modal=1))
        vis_labels.append(pid)

    for image_path in ir_images:
        ir_features.append(extract_embedding(image_path, modal=2))
        ir_labels.append(pid)

if len(vis_features) == 0 or len(ir_features) == 0:
    raise RuntimeError("没有成功提取到特征，请检查路径、身份编号和相机目录。")

vis_features = np.stack(vis_features, axis=0).astype(np.float32)
ir_features = np.stack(ir_features, axis=0).astype(np.float32)
vis_labels = np.asarray(vis_labels, dtype=np.int64)
ir_labels = np.asarray(ir_labels, dtype=np.int64)

print("VIS features:", vis_features.shape)
print("IR features:", ir_features.shape)

# ============================================================
# 8. 计算跨模态余弦距离
# ============================================================
similarity_matrix = np.matmul(vis_features, ir_features.T)
similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
distance_matrix = 1.0 - similarity_matrix

same_id_mask = vis_labels[:, None] == ir_labels[None, :]
intra_distances = distance_matrix[same_id_mask]
inter_distances = distance_matrix[~same_id_mask]

intra_distances = intra_distances[np.isfinite(intra_distances)]
inter_distances = inter_distances[np.isfinite(inter_distances)]

if len(intra_distances) == 0:
    raise RuntimeError("没有得到类内跨模态样本对。")
if len(inter_distances) == 0:
    raise RuntimeError("没有得到类间跨模态样本对。")

mean_intra = float(np.mean(intra_distances))
mean_inter = float(np.mean(inter_distances))
delta_s = mean_inter - mean_intra

print("Intra pairs:", len(intra_distances))
print("Inter pairs:", len(inter_distances))
print(f"Intra mean: {mean_intra:.4f}")
print(f"Inter mean: {mean_inter:.4f}")
print(f"Delta_s: {delta_s:.4f}")

# ============================================================
# 9. 绘制分布图
# ============================================================
all_distances = np.concatenate([intra_distances, inter_distances])

x_left = max(0.0, float(np.percentile(all_distances, 0.5)) - 0.03)
x_right = min(2.0, float(np.percentile(all_distances, 99.5)) + 0.03)
x_left = min(x_left, mean_intra - 0.03)
x_right = max(x_right, mean_inter + 0.03)

bin_edges = np.linspace(x_left, x_right, NUM_BINS + 1)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 15,
    "mathtext.fontset": "stix",
})

fig, ax = plt.subplots(figsize=(8.0, 6.3), dpi=150)

intra_color = "#6464F4"
inter_color = "#63AD60"
intra_line_color = "#2F36E8"
inter_line_color = "#24892E"

ax.hist(
    intra_distances,
    bins=bin_edges,
    density=True,
    histtype="stepfilled",
    alpha=0.72,
    color=intra_color,
    edgecolor=intra_color,
    linewidth=0.35,
    label="Intra-class",
    zorder=2
)

ax.hist(
    inter_distances,
    bins=bin_edges,
    density=True,
    histtype="stepfilled",
    alpha=0.72,
    color=inter_color,
    edgecolor=inter_color,
    linewidth=0.35,
    label="Inter-class",
    zorder=1
)

ax.axvline(
    mean_intra,
    color=intra_line_color,
    linestyle="--",
    linewidth=1.7,
    alpha=0.95,
    zorder=5
)
ax.axvline(
    mean_inter,
    color=inter_line_color,
    linestyle="--",
    linewidth=1.7,
    alpha=0.95,
    zorder=5
)

_, original_y_max = ax.get_ylim()
ax.set_ylim(0.0, original_y_max * 1.16)
y_max = ax.get_ylim()[1]
x_span = x_right - x_left

mean_text_y = y_max * 0.58
ax.text(
    mean_intra + x_span * 0.012,
    mean_text_y,
    f"{mean_intra:.4f}",
    color=intra_line_color,
    ha="left",
    va="center",
    fontsize=14,
    fontweight="bold",
    zorder=7
)
ax.text(
    mean_inter - x_span * 0.012,
    mean_text_y,
    f"{mean_inter:.4f}",
    color=inter_line_color,
    ha="right",
    va="center",
    fontsize=14,
    fontweight="bold",
    zorder=7
)

arrow_y = y_max * 0.83
delta_text_y = y_max * 0.865
mid_x = (mean_intra + mean_inter) / 2.0

ax.annotate(
    "",
    xy=(mean_inter, arrow_y),
    xytext=(mean_intra, arrow_y),
    arrowprops=dict(
        arrowstyle="<->",
        color="black",
        linewidth=1.8,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=13
    ),
    zorder=7
)

ax.text(
    mid_x,
    delta_text_y,
    rf"$\delta_s={delta_s:.4f}$",
    color="black",
    ha="center",
    va="bottom",
    fontsize=18,
    fontweight="bold",
    zorder=7
)

ax.set_xlim(x_left, x_right)
ax.set_xlabel("Cosine Distance")
# 为复刻 DEEN 图保留 Frequency；density=True 时严格含义为 Density。
ax.set_ylabel("Frequency")

legend = ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    borderpad=0.55
)
legend.get_frame().set_edgecolor("#CCCCCC")
legend.get_frame().set_linewidth(1.0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)
ax.tick_params(axis="both", direction="out", length=4, width=1.0)
ax.grid(False)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)

pdf_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + ".pdf")
png_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + ".png")

plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Saved PDF:", pdf_path)
print("Saved PNG:", png_path)
