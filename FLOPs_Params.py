import torch
import torch.nn as nn
from thop import profile


# ==========================================
# 1. 导入或定义你的模型
# ==========================================
# 假设你的模型定义在 model.py 中，类名为 embed_net
from model import embed_net

# 如果你把这段代码直接粘贴在模型定义文件里，就不需要上面的 import
# 这里假设 'embed_net' 类已经存在于内存中

# ==========================================
# 2. 定义包装器 (Wrapper) - 关键步骤
# ==========================================
class ReIDModelWrapper(nn.Module):
    """
    这是一个包装器，用于告诉 thop 如何进行单张图片的推理。
    如果不使用这个，thop 可能会进入 modal=0 模式，导致计算量翻倍。
    """

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        # 我们模拟单张图片推理 (Single-shot Inference)
        # 强制 modal=1 (Visible mode) 或 modal=2 (Thermal mode)
        # 这样只会激活其中一个分支，符合 ReID 推理的实际情况

        # 注意：虽然 modal=1 时 ir 参数会被忽略，但为了满足函数签名，
        # 我们还是把 x 传给 ir，或者传 None (取决于你的 visible_module 是否完全独立)
        return self.model(x1=x, x2=x, modal=2)


# ==========================================
# 3. 初始化模型和数据
# ==========================================
# 实例化你的模型，class_num 可以随意填，因为最后一层全连接层的差异对整体 Params/FLOPs 影响微乎其微
# 确保这里的参数与你训练/测试时的一致 (例如 arch='resnet50')
model = embed_net(class_num=395, dataset="sysu", arch='resnet50')

# 切换到评估模式 (非常重要！这会冻结 BN 层并影响某些层的行为)
model.eval()

# 使用包装器包裹模型
wrapped_model = ReIDModelWrapper(model)

# 定义输入张量 (Batch Size = 1)
# 你的指定尺寸: (1, 3, 384, 144)
input_tensor = torch.randn(1, 3, 384, 144)

# ==========================================
# 4. 计算 Params 和 FLOPs
# ==========================================
print(f"正在计算 Params 和 FLOPs...")
print(f"输入尺寸: {input_tensor.shape}")
print(f"测试模式: modal=1 (单模态推理)")

# thop.profile 会执行一次前向传播来追踪运算
# custom_ops 参数用于处理某些自定义层，通常留空即可
flops, params = profile(wrapped_model, inputs=(input_tensor,), verbose=False)

# ==========================================
# 5. 格式化输出
# ==========================================
flops_g = flops / 1e9  # 转换为 G (Giga FLOPs)
params_m = params / 1e6  # 转换为 M (Million Params)

print("-" * 30)
print(f"FLOPs:  {flops_g:.3f} G")
print(f"Params: {params_m:.3f} M")
print("-" * 30)

# 额外检查：如果你想看更详细的每层消耗，可以使用 thop.clever_format
from thop import clever_format

flops_str, params_str = clever_format([flops, params], "%.3f")
print(f"格式化输出: FLOPs {flops_str}, Params {params_str}")