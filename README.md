# SMILEtrack_Official
SMILEtrack: SiMIlarity LEarning for Occlusion-Aware Multiple Object Tracking

## 新增功能

### 1. 自适应图像分块 (AdaptivePatches)

`models/patches.py` 中实现了一个灵活的图像分块策略，可以将输入特征图划分为重叠或不重叠的小块，主要用于目标跟踪中的局部特征提取和相似度计算。

#### 使用方法
```python
from models.patches import AdaptivePatches

# 创建分块模块实例
patches_module = AdaptivePatches(patch_size=16, stride=8)

# 输入特征图处理
x = torch.randn(1, 64, 128, 128)  # 示例输入
patches = patches_module(x)  # 获取分块结果
merged = patches_module.merge_patches(patches, (128, 128))  # 合并回原始尺寸
```

### 2. Grad-CAM 可视化 (GradCAM)

`utils/grad_cam.py` 实现了Grad-CAM可视化技术，用于解释深度神经网络的决策过程，通过计算目标类别相对于特定层的梯度，生成类激活热力图，显示模型在做出预测时关注的图像区域。

#### 使用方法
```python
from utils.grad_cam import GradCAM, visualize_cam

# 创建GradCAM实例
model = YourModel()
target_layer = model.layer4  # 使用最后一个卷积层
grad_cam = GradCAM(model, target_layer)

# 生成并可视化CAM
image = preprocess_image(image)  # 预处理输入图像
cam = grad_cam(image)  # 生成CAM
visualization = visualize_cam(original_image, cam)  # 可视化结果
```
