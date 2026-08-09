import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

########################################
# 配置区（你可以修改）
########################################

# 模型与数据路径
MODEL_PATH = ""
SYSU_PATH = "./Datasets/SYSU-MM01/"

selected_ids = [89,90,93,102,105,108,112,116,117,122,125,129,130,134,138,139,150,152,162,166]

# 每个模态下的图片数
K = 10

# 摄像头定义
vis_cams = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cams  = ['cam3', 'cam6']

# 图像输入大小
IMG_W = 144
IMG_H = 384

########################################
# 预处理
########################################
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

########################################
# 加载模型
########################################
from model import embed_net  # 直接复用你的工程

device = 'cuda:0'
n_class = 395  # 对于 SYSU-MM01 固定
net = embed_net(n_class, dataset='sysu', arch='resnet50')
net.to(device)
cudnn.benchmark = True

print("Loading model:", MODEL_PATH)
checkpoint = torch.load(MODEL_PATH, weights_only=False)
net.load_state_dict(checkpoint["net"])
net.eval()

def load_images_for_id(pid, cams, base_path):
    img_list_per_cam = {}

    for cam in cams:
        cam_dir = os.path.join(base_path, cam, "{:04d}".format(pid))
        if not os.path.isdir(cam_dir):
            continue
        images = sorted([os.path.join(cam_dir, x)
                         for x in os.listdir(cam_dir)
                         if x.lower().endswith(".jpg")])
        if len(images) > 0:
            img_list_per_cam[cam] = images

    return img_list_per_cam


def pick_evenly_from_cams(img_lists, K):
    """
    img_lists: {cam1:[], cam2:[], ...}
    """
    cams = list(img_lists.keys())
    C = len(cams)

    if C == 0:
        return []

    base = K // C         
    remain = K - base * C 
    selected = []

    for idx, cam in enumerate(cams):
        take = base + (1 if idx < remain else 0)
        imgs = img_lists[cam][:take] 
        selected.extend(imgs)

    return selected


def extract_embedding(img_path,modal=1):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = transform_test(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        feat, feat_att = net(img, img, modal=modal)  # modal=1 对应 VIS/IR input 均可

    # feat = [3,2048]
    # feat_att = [3,2048]
    f1 = feat.mean(dim=0)         # [2048]
    f2 = feat_att.mean(dim=0)     # [2048]

    a = 0.5
    final_feat = a*f1 + (1-a)*f2

    return final_feat.cpu().numpy()


########################################
# 颜色表（固定 20 种强区分颜色）
########################################
colors = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe",
    "#008080","#e6beff","#9a6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#808080"
]

########################################
# 主流程 — 收集所有要参加 t-SNE 的 embedding
########################################
all_feats = []
all_labels = []
all_modals = []  # VIS=1 IR=0

for idx, pid in enumerate(selected_ids):
    print(f"Processing ID {pid}")

    # VIS
    vis_lists = load_images_for_id(pid, vis_cams, SYSU_PATH)
    vis_imgs = pick_evenly_from_cams(vis_lists, K)

    # IR
    ir_lists = load_images_for_id(pid, ir_cams, SYSU_PATH)
    ir_imgs = pick_evenly_from_cams(ir_lists, K)

    # 提取 embedding
    for path in vis_imgs:
        feat = extract_embedding(path,modal=1)
        all_feats.append(feat)
        all_labels.append(idx)
        all_modals.append(1)  # VIS -> X

    for path in ir_imgs:
        feat = extract_embedding(path,modal=2)
        all_feats.append(feat)
        all_labels.append(idx)
        all_modals.append(0)  # IR -> O

all_feats = np.array(all_feats)      # [N,2048]
all_labels = np.array(all_labels)    # [N]
all_modals = np.array(all_modals)    # [N]


########################################
# t-SNE 可视化
########################################
print("Running t-SNE...")
X_2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(all_feats)

plt.figure(figsize=(12, 10), dpi=120)

for i in range(len(X_2d)):
    cid = all_labels[i]
    modal = all_modals[i]

    if modal == 1:
        marker = '^'  # VIS
    else:
        marker = 'o'  # IR

    plt.scatter(X_2d[i,0], X_2d[i,1],
                color=colors[cid],
                marker=marker,
                s=80,
                edgecolors='black',
                linewidths=0.5)

# plt.title("t-SNE of VIS/IR Embeddings (20 IDs, 10 img each modality)")
plt.grid(False)
plt.tight_layout()
plt.savefig("tsne_sysu.png", dpi=300)
plt.show()
