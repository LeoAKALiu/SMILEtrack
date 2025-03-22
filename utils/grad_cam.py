import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """Grad-CAM (Gradient-weighted Class Activation Mapping) 实现
    
    该类实现了Grad-CAM可视化技术，用于解释深度神经网络的决策过程。
    通过计算目标类别相对于特定层的梯度，生成类激活热力图，
    显示模型在做出预测时关注的图像区域。
    
    参数:
        model: 需要可视化的PyTorch模型
        target_layer: 用于生成CAM的目标层（通常是最后的卷积层）
    
    使用示例:
        >>> model = YourModel()
        >>> target_layer = model.layer4  # 假设使用最后一个卷积层
        >>> grad_cam = GradCAM(model, target_layer)
        >>> image = preprocess_image(image)  # 预处理输入图像
        >>> cam = grad_cam(image)  # 生成CAM
        >>> visualization = visualize_cam(original_image, cam)  # 可视化结果
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # 注册钩子
        target_layer.register_forward_hook(self.save_features)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        self.features = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, index=None):
        # 前向传播
        output = self.model(x)
        
        if index is None:
            index = torch.argmax(output)
        
        # 清除梯度
        self.model.zero_grad()
        
        # 反向传播
        output[0, index].backward()
        
        # 获取特征图和梯度
        gradients = self.gradients
        features = self.features
        
        # 计算权重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 生成cam
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 归一化
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.squeeze().cpu().numpy()

def visualize_cam(img, cam):
    """将CAM叠加到原图上"""
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    result = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
    return result