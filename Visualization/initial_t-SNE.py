import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
import torchvision.transforms as transforms

########################################
# 配置区
########################################

SYSU_PATH = "./Datasets/SYSU-MM01/"

selected_ids = [
    89, 90, 93, 102, 105, 108, 112, 116, 117, 122,
    125, 129, 130, 134, 138, 139, 150, 152, 162, 166
]

K = 10

vis_cams = ["cam1", "cam2", "cam4", "cam5"]
ir_cams = ["cam3", "cam6"]

# 原始像素降采样尺寸
RAW_W = 48
RAW_H = 128

PCA_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 0

OUTPUT_PATH = "initial.png"

########################################
# 原始图片预处理
########################################

transform_raw = transforms.Compose([
    transforms.Resize((RAW_H, RAW_W)),
    transforms.ToTensor(),
])

########################################
# 加载某个身份在指定摄像头下的图片
########################################

def load_images_for_id(pid, cams, base_path):
    img_list_per_cam = {}

    for cam in cams:
        cam_dir = os.path.join(base_path, cam, f"{pid:04d}")

        if not os.path.isdir(cam_dir):
            continue

        images = sorted([
            os.path.join(cam_dir, filename)
            for filename in os.listdir(cam_dir)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        if images:
            img_list_per_cam[cam] = images

    return img_list_per_cam

########################################
# 按摄像头均匀选择 K 张图片
########################################

def pick_evenly_from_cams(img_lists, k):
    cams = list(img_lists.keys())
    cam_num = len(cams)

    if cam_num == 0:
        return []

    base = k // cam_num
    remain = k - base * cam_num

    selected = []

    for idx, cam in enumerate(cams):
        take = base + (1 if idx < remain else 0)
        selected.extend(img_lists[cam][:take])

    return selected

########################################
# 提取原始像素向量
########################################

def extract_raw_pixels(img_path):
    """
    完全不经过神经网络：
    RGB图片 -> resize -> [0,1]像素 -> 展平 -> L2归一化
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        tensor = transform_raw(img)

    feat = tensor.numpy().reshape(-1).astype(np.float32)
    feat /= np.linalg.norm(feat) + 1e-12

    return feat

########################################
# 固定颜色
########################################

colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"
]

########################################
# 收集原始像素
########################################

all_feats = []
all_labels = []
all_modals = []

for idx, pid in enumerate(selected_ids):
    print(f"Processing ID {pid}")

    vis_lists = load_images_for_id(pid, vis_cams, SYSU_PATH)
    vis_imgs = pick_evenly_from_cams(vis_lists, K)

    ir_lists = load_images_for_id(pid, ir_cams, SYSU_PATH)
    ir_imgs = pick_evenly_from_cams(ir_lists, K)

    print(f"  VIS={len(vis_imgs)}, IR={len(ir_imgs)}")

    for path in vis_imgs:
        all_feats.append(extract_raw_pixels(path))
        all_labels.append(idx)
        all_modals.append(1)

    for path in ir_imgs:
        all_feats.append(extract_raw_pixels(path))
        all_labels.append(idx)
        all_modals.append(0)

if len(all_feats) == 0:
    raise RuntimeError("No images loaded. Check SYSU_PATH and selected_ids.")

all_feats = np.asarray(all_feats, dtype=np.float32)
all_labels = np.asarray(all_labels, dtype=np.int64)
all_modals = np.asarray(all_modals, dtype=np.int64)

print("Raw pixel feature shape:", all_feats.shape)

########################################
# PCA
########################################

pca_dim = min(PCA_DIM, all_feats.shape[0] - 1, all_feats.shape[1])

if pca_dim < 2:
    raise RuntimeError("Too few samples for PCA and t-SNE.")

print(f"Running PCA: {all_feats.shape[1]} -> {pca_dim}")

pca = PCA(
    n_components=pca_dim,
    svd_solver="randomized",
    random_state=TSNE_RANDOM_STATE
)

all_feats_pca = pca.fit_transform(all_feats)

print(
    "PCA explained variance ratio:",
    f"{pca.explained_variance_ratio_.sum():.4f}"
)

########################################
# t-SNE
########################################

sample_num = len(all_feats_pca)
perplexity = min(TSNE_PERPLEXITY, sample_num - 1)

print(
    f"Running t-SNE: samples={sample_num}, perplexity={perplexity}"
)

X_2d = TSNE(
    n_components=2,
    perplexity=perplexity,
    init="pca",
    learning_rate="auto",
    random_state=TSNE_RANDOM_STATE
).fit_transform(all_feats_pca)

########################################
# 绘图
########################################

plt.figure(figsize=(12, 10), dpi=120)

for i in range(len(X_2d)):
    cid = all_labels[i]
    modal = all_modals[i]

    marker = "^" if modal == 1 else "o"

    plt.scatter(
        X_2d[i, 0],
        X_2d[i, 1],
        color=colors[cid],
        marker=marker,
        s=80,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9
    )

plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02
)

print("Saved:", OUTPUT_PATH)
plt.show()
