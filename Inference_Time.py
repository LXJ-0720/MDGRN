import torch
import time
import numpy as np

# ================= 配置区域 =================
# 1. 导入你的模型类 (假设在当前脚本或已 import)
from model import embed_net

# 假设我们在测试 SYSU-MM01 的输入尺寸 (根据你的要求)
INPUT_SHAPE = (1, 3, 384, 144)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 准备工作 =================

# 1. 实例化模型 (使用随机权重即可)
# class_num 随便填，gm_pool 保持和你实验一致，arch 保持一致
print("正在初始化模型...")
model = embed_net(class_num=395,  dataset="sysu", arch='resnet50')

# 2. 转移到 GPU 并开启评估模式
model.to(device)
model.eval()

# 3. 生成虚拟数据 (Dummy Input)
# 只需要一个张量，传给 vis 和 ir 都可以，因为 modal=1 时只用其中一个
dummy_input = torch.randn(INPUT_SHAPE).to(device)

# 4. 开启 cudnn benchmark
# 对于输入尺寸固定的网络，开启这个可以让 cudnn 寻找最快的卷积算法，
# 这更符合实际部署时的最佳性能。
torch.backends.cudnn.benchmark = True

# ================= 测量流程 =================

# --- A. 预热 (Warm-up) ---
# GPU 刚开始工作时需要显存分配和初始化，第一次运行通常很慢。
# 必须先空跑几圈，让 GPU 进入“高性能状态”。
print("正在预热 GPU (Warm-up)...")
with torch.no_grad():
    for _ in range(1000):
        # 模拟 modal=1 (可见光分支)
        _ = model(x1=dummy_input, x2=dummy_input, modal=1)

# 确保预热完成
torch.cuda.synchronize()

# --- B. 正式计时 (Testing) ---
# 循环运行多次取平均，消除抖动
iterations = 10000  # 建议跑 1000 次
timings = []  # 记录每次的时间

print(f"开始测量 Inference Time (循环 {iterations} 次)...")

with torch.no_grad():
    for i in range(iterations):
        # 1. 计时起点同步
        torch.cuda.synchronize()
        start_time = time.time()

        # 2. 模型推理 (核心部分)
        # 这里的输入数据已经在 GPU 上了，所以不包含 I/O 时间
        _ = model(x1=dummy_input, x2=dummy_input, modal=1)

        # 3. 计时终点同步
        torch.cuda.synchronize()
        end_time = time.time()

        # 记录单次耗时 (单位：毫秒 ms)
        timings.append((end_time - start_time) * 1000)

# ================= 结果统计 =================
avg_time = np.mean(timings)
std_time = np.std(timings)

print("-" * 30)
print(f"Model: embed_net (ResNet50 backbone)")
print(f"Input Shape: {INPUT_SHAPE}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print("-" * 30)
print(f"Average Inference Time: {avg_time:.4f} ms")
print(f"Standard Deviation:     {std_time:.4f} ms")
print("-" * 30)

# 估算 FPS (Frames Per Second)
print(f"Estimated FPS: {1000 / avg_time:.2f}")