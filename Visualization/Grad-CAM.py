import os
from typing import Dict, List, Tuple

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from pytorch_grad_cam.utils.image import show_cam_on_image

from model import embed_net


# ============================================================
# 1. 配置区
# ============================================================

MODEL_TAG = "Ours"
# MODEL_TAG = "DEEN"

MODEL_CHECKPOINT = (
    ""
)

DATA_ROOT = ""
OUTPUT_ROOT = "save_gradcam_reference_style"

TARGET_IDS = [10]
CAMS_LIST = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]

# None 表示处理该摄像头下全部图片
MAX_IMAGES_PER_CAMERA = 5

NUM_CLASSES = 395
IMG_W = 144
IMG_H = 384
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "sysu"
ARCH_NAME = "resnet50"

# 当前模型中的属性名
DEE_ATTR = "DEE"
MFA2_ATTR = "MFA2"
CNL_ATTR = "CNL"
PNL_ATTR = "PNL"

SAVE_MSDEE = True
SAVE_DGCLRM = True
SAVE_CNL = True
SAVE_PNL = True

CAM_GAMMA = 0.70

# 热力图和原图的叠加强度
IMAGE_WEIGHT = 0.50

# 是否打印每张图片三个分支各自的预测类别
PRINT_PREDICTIONS = True


# ============================================================
# 2. 基础设置
# ============================================================

device = torch.device(DEVICE)
cudnn.benchmark = True

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

transform_test = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN.tolist(),
        std=STD.tolist(),
    ),
])


# ============================================================
# 3. 文件与图像辅助函数
# ============================================================

def ensure_dir_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )


def list_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []

    image_paths = sorted(
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if is_image_file(filename)
    )

    if MAX_IMAGES_PER_CAMERA is not None:
        image_paths = image_paths[:MAX_IMAGES_PER_CAMERA]

    return image_paths


def determine_modal(cam_name: str) -> int:
    # SYSU-MM01: cam3、cam6 为红外，其余为可见光
    return 2 if cam_name.lower() in {"cam3", "cam6"} else 1


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * STD[None, None, :] + MEAN[None, None, :]
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def resize_map(
    response: np.ndarray,
    output_hw: Tuple[int, int],
) -> np.ndarray:
    target_h, target_w = output_hw

    if response.shape == (target_h, target_w):
        return response.astype(np.float32)

    response_tensor = torch.from_numpy(
        np.asarray(response, dtype=np.float32)
    )[None, None]

    response_tensor = F.interpolate(
        response_tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    return response_tensor[0, 0].cpu().numpy().astype(np.float32)


def minmax_normalize(
    response: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    response = np.asarray(response, dtype=np.float32)
    response = np.maximum(response, 0.0)

    minimum = float(response.min())
    maximum = float(response.max())

    if maximum - minimum < 1e-12:
        return np.zeros_like(response, dtype=np.float32)

    response = (response - minimum) / (maximum - minimum)

    if gamma != 1.0:
        response = np.power(
            np.clip(response, 0.0, 1.0),
            gamma,
        )

    return response.astype(np.float32)


def save_input(
    rgb_image: np.ndarray,
    save_path: str,
) -> None:
    ensure_dir_exists(os.path.dirname(save_path))

    image_uint8 = np.clip(
        np.round(rgb_image * 255.0),
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(image_uint8).save(save_path)


def save_overlay(
    rgb_image: np.ndarray,
    cam_map: np.ndarray,
    save_path: str,
) -> None:
    cam_map = resize_map(
        cam_map,
        rgb_image.shape[:2],
    )
    cam_map = np.clip(cam_map, 0.0, 1.0)

    # 新版 grad-cam 支持 image_weight；旧版不支持时自动回退
    try:
        visualization = show_cam_on_image(
            rgb_image,
            cam_map,
            use_rgb=True,
            image_weight=IMAGE_WEIGHT,
        )
    except TypeError:
        visualization = show_cam_on_image(
            rgb_image,
            cam_map,
            use_rgb=True,
        )

    ensure_dir_exists(os.path.dirname(save_path))
    Image.fromarray(visualization).save(save_path)


# ============================================================
# 4. 加载模型
# ============================================================

print("==> Building model...")
model = embed_net(
    NUM_CLASSES,
    dataset=DATASET_NAME,
    arch=ARCH_NAME,
).to(device)

if not os.path.isfile(MODEL_CHECKPOINT):
    raise FileNotFoundError(
        f"找不到模型权重：{MODEL_CHECKPOINT}"
    )

print("==> Loading checkpoint:", MODEL_CHECKPOINT)
checkpoint = torch.load(
    MODEL_CHECKPOINT,
    map_location=device,
    weights_only=False,
)

state_dict = (
    checkpoint["net"]
    if isinstance(checkpoint, dict) and "net" in checkpoint
    else checkpoint
)

if isinstance(state_dict, dict) and state_dict:
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

model.load_state_dict(state_dict, strict=True)

if isinstance(checkpoint, dict) and "epoch" in checkpoint:
    print("==> Loaded epoch:", checkpoint["epoch"])

# 需要训练分支返回 classifier logits；
# 同时将 BN / Dropout 固定为 eval，避免单张图片改变统计量。
model.train()

for module in model.modules():
    if isinstance(
        module,
        (
            nn.modules.batchnorm._BatchNorm,
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
        ),
    ):
        module.eval()


# ============================================================
# 5. 注册需要可视化的模块
# ============================================================

try:
    final_target_layer = model.base_resnet.base.layer4[-1]
except AttributeError as error:
    raise AttributeError(
        "找不到 model.base_resnet.base.layer4[-1]"
    ) from error

target_modules: Dict[str, nn.Module] = {
    "overall": final_target_layer,
}

if SAVE_MSDEE:
    if not hasattr(model, DEE_ATTR):
        raise AttributeError(
            f"当前模型找不到 model.{DEE_ATTR}"
        )
    target_modules["msdee"] = getattr(model, DEE_ATTR)

if SAVE_DGCLRM:
    if not hasattr(model, MFA2_ATTR):
        raise AttributeError(
            f"当前模型找不到 model.{MFA2_ATTR}"
        )
    target_modules["dgclrm"] = getattr(model, MFA2_ATTR)

if SAVE_CNL:
    mfa2_module = getattr(model, MFA2_ATTR)
    if not hasattr(mfa2_module, CNL_ATTR):
        raise AttributeError(
            f"当前模型找不到 model.{MFA2_ATTR}.{CNL_ATTR}"
        )
    target_modules["cnl"] = getattr(mfa2_module, CNL_ATTR)

if SAVE_PNL:
    mfa2_module = getattr(model, MFA2_ATTR)
    if not hasattr(mfa2_module, PNL_ATTR):
        raise AttributeError(
            f"当前模型找不到 model.{MFA2_ATTR}.{PNL_ATTR}"
        )
    target_modules["pnl"] = getattr(mfa2_module, PNL_ATTR)


# ============================================================
# 6. Grad-CAM 核心
# ============================================================

def extract_logits(outputs) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(
            f"训练模式下模型应返回 tuple/list，实际为 {type(outputs)}"
        )

    if len(outputs) < 2:
        raise ValueError(
            f"模型输出不足两项，实际长度为 {len(outputs)}"
        )

    logits = outputs[1]

    if not torch.is_tensor(logits) or logits.dim() != 2:
        raise ValueError(
            "分类 logits 应为二维 Tensor，"
            f"实际为 {getattr(logits, 'shape', None)}"
        )

    return logits


def select_4d_tensor(
    output,
    module_name: str,
) -> torch.Tensor:
    if torch.is_tensor(output) and output.dim() == 4:
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.dim() == 4:
                return item

    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item) and item.dim() == 4:
                return item

    raise TypeError(
        f"{module_name} 的输出中找不到 4D Tensor，"
        f"实际类型为 {type(output)}"
    )


def raw_gradcam(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    output_hw: Tuple[int, int],
) -> np.ndarray:
    """
    activation / gradient: [C,H,W]
    """
    channel_weights = gradient.mean(
        dim=(1, 2),
        keepdim=True,
    )

    cam_map = (
        channel_weights * activation
    ).sum(dim=0)

    cam_map = F.relu(cam_map)

    cam_map = F.interpolate(
        cam_map[None, None],
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return cam_map.detach().cpu().numpy().astype(np.float32)


def choose_feature_index(
    feature_batch: int,
    logit_index: int,
    logit_batch: int,
) -> int:
    """
    layer4 / MSDEE 常见为 3 个样本；
    DG-CLRM / CNL / PNL 常见为 1 个共享样本。
    """
    if feature_batch == logit_batch:
        return logit_index

    if feature_batch == 1:
        return 0

    return min(logit_index, feature_batch - 1)


@torch.enable_grad()
def generate_maps(
    input_tensor: torch.Tensor,
    modal: int,
) -> Dict[str, np.ndarray]:

    model.zero_grad(set_to_none=True)

    activations: Dict[str, torch.Tensor] = {}
    handles = []

    def register_hook(
        name: str,
        module: nn.Module,
    ) -> None:
        def forward_hook(_module, _inputs, output):
            activations[name] = select_4d_tensor(
                output,
                name,
            )

        handles.append(
            module.register_forward_hook(forward_hook)
        )

    for name, module in target_modules.items():
        register_hook(name, module)

    try:
        outputs = model(
            input_tensor,
            input_tensor,
            modal=modal,
        )

        logits = extract_logits(outputs)
        predictions = logits.argmax(dim=1)

        if PRINT_PREDICTIONS:
            print(
                "branch predictions =",
                predictions.detach().cpu().tolist(),
            )

        output_hw = tuple(input_tensor.shape[-2:])
        results: Dict[str, np.ndarray] = {}

        # 参考代码风格：
        # 每个最终分支都使用它自己的预测类别；
        # 每张 CAM 先归一化，再平均。
        for module_name, activation in activations.items():
            normalized_branch_cams = []

            for logit_index in range(logits.size(0)):
                target_class = int(
                    predictions[logit_index].item()
                )

                score = logits[
                    logit_index,
                    target_class,
                ]

                gradient = torch.autograd.grad(
                    score,
                    activation,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]

                if gradient is None:
                    raise RuntimeError(
                        f"{module_name} 没有获得梯度，"
                        "请检查该模块输出是否进入分类路径。"
                    )

                feature_index = choose_feature_index(
                    feature_batch=activation.size(0),
                    logit_index=logit_index,
                    logit_batch=logits.size(0),
                )

                raw_cam = raw_gradcam(
                    activation[feature_index],
                    gradient[feature_index],
                    output_hw,
                )

                # 每一分支单独归一化和 gamma 增强
                normalized_branch_cam = minmax_normalize(
                    raw_cam,
                    gamma=CAM_GAMMA,
                )

                normalized_branch_cams.append(
                    normalized_branch_cam
                )

            # 单独归一化后的多分支 CAM 再平均
            fused_cam = np.mean(
                np.stack(
                    normalized_branch_cams,
                    axis=0,
                ),
                axis=0,
            )

            # 最后再统一拉伸一次
            results[module_name] = minmax_normalize(
                fused_cam,
                gamma=1.0,
            )

        return results

    finally:
        for handle in handles:
            handle.remove()


# ============================================================
# 7. 单张图片处理
# ============================================================

def process_one_image(
    image_path: str,
    cam_name: str,
    pid_folder: str,
) -> None:

    image_pil = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform_test(image_pil)
    input_tensor = image_tensor.unsqueeze(0).to(device)

    rgb_image = tensor_to_rgb_image(image_tensor)
    modal = determine_modal(cam_name)

    response_maps = generate_maps(
        input_tensor,
        modal,
    )

    filename = os.path.basename(image_path)
    save_root = os.path.join(
        OUTPUT_ROOT,
        MODEL_TAG,
    )

    input_path = os.path.join(
        save_root,
        "input",
        cam_name,
        pid_folder,
        filename,
    )
    save_input(rgb_image, input_path)

    for map_name, cam_map in response_maps.items():
        save_path = os.path.join(
            save_root,
            map_name,
            cam_name,
            pid_folder,
            filename,
        )
        save_overlay(
            rgb_image,
            cam_map,
            save_path,
        )


# ============================================================
# 8. 主程序
# ============================================================

def main() -> None:
    total = 0
    failed = 0

    print("MODEL_TAG :", MODEL_TAG)
    print("SAVE ROOT :", os.path.join(OUTPUT_ROOT, MODEL_TAG))
    print("MAP NAMES :", list(target_modules.keys()))
    print("CAM_GAMMA :", CAM_GAMMA)

    for pid in TARGET_IDS:
        pid_folder = f"{int(pid):04d}"

        for cam_name in CAMS_LIST:
            source_dir = os.path.join(
                DATA_ROOT,
                cam_name,
                pid_folder,
            )

            image_paths = list_image_files(source_dir)

            if not image_paths:
                continue

            print(
                f"Processing ID={pid}, "
                f"camera={cam_name}, "
                f"images={len(image_paths)}"
            )

            for image_path in tqdm(
                image_paths,
                desc=f"{cam_name}-{pid_folder}",
            ):
                total += 1

                try:
                    process_one_image(
                        image_path,
                        cam_name,
                        pid_folder,
                    )
                except Exception as error:
                    failed += 1
                    print("\n处理失败：", image_path)
                    print("错误信息：", repr(error))

    print("\nFinished")
    print("Total :", total)
    print("Failed:", failed)
    print(
        "Saved :",
        os.path.join(OUTPUT_ROOT, MODEL_TAG),
    )


if __name__ == "__main__":
    main()
