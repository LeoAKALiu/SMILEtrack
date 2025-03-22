import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePatches(nn.Module):
    """自适应图像分块模块
    
    该模块实现了一个灵活的图像分块策略，可以将输入特征图划分为不同布局的小块。
    支持多种分块布局模式，包括垂直分块、水平分块、中心重叠分块等。
    主要用于目标跟踪中的局部特征提取和相似度计算。
    
    参数:
        layout_type (str): 分块布局类型，可选值为['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            A: 水平两等分
            B: 垂直两等分
            C: 水平三等分
            D: 中心垂直分块
            E: 中心十字分块
            F: 四等分
            G: 中心重叠分块
            H: 中心水平分块
            I: 四肢分块
    
    使用示例:
        >>> patches_module = AdaptivePatches(layout_type='E')
        >>> x = torch.randn(1, 64, 128, 128)  # 输入特征图
        >>> patches = patches_module(x)  # 获取分块结果
        >>> merged = patches_module.merge_patches(patches, (128, 128))  # 合并回原始尺寸
    """
    def __init__(self, layout_type='E'):
        super(AdaptivePatches, self).__init__()
        self.layout_type = layout_type
        
    def layout_generator(self, H, W):
        """生成不同布局的分块坐标
        Args:
            H: 特征图高度
            W: 特征图宽度
        Returns:
            coords: 分块坐标列表 [(y1,x1,y2,x2), ...]
        """
        coords = []
        if self.layout_type == 'A':  # 水平两等分
            h_mid = H // 2
            coords = [(0, 0, h_mid, W), (h_mid, 0, H, W)]
            
        elif self.layout_type == 'B':  # 垂直两等分
            w_mid = W // 2
            coords = [(0, 0, H, w_mid), (0, w_mid, H, W)]
            
        elif self.layout_type == 'C':  # 水平三等分
            h1, h2 = H // 3, 2 * H // 3
            coords = [(0, 0, h1, W), (h1, 0, h2, W), (h2, 0, H, W)]
            
        elif self.layout_type == 'D':  # 中心垂直分块
            h_mid, w_mid = H // 2, W // 2
            w_quarter = W // 4
            coords = [(0, w_quarter, H, 3*w_quarter)]
            
        elif self.layout_type == 'E':  # 中心十字分块
            h_mid, w_mid = H // 2, W // 2
            h_quarter, w_quarter = H // 4, W // 4
            coords = [
                (h_quarter, 0, 3*h_quarter, W),  # 水平条
                (0, w_quarter, H, 3*w_quarter)   # 垂直条
            ]
            
        elif self.layout_type == 'F':  # 四等分
            h_mid, w_mid = H // 2, W // 2
            coords = [
                (0, 0, h_mid, w_mid),        # 左上
                (0, w_mid, h_mid, W),         # 右上
                (h_mid, 0, H, w_mid),         # 左下
                (h_mid, w_mid, H, W)          # 右下
            ]
            
        elif self.layout_type == 'G':  # 中心重叠分块
            h_mid, w_mid = H // 2, W // 2
            h_quarter, w_quarter = H // 4, W // 4
            coords = [
                (h_quarter, w_quarter, 3*h_quarter, 3*w_quarter),  # 中心
                (0, w_quarter, H, 3*w_quarter)                      # 垂直条
            ]
            
        elif self.layout_type == 'H':  # 中心水平分块
            h_mid = H // 2
            h_quarter = H // 4
            coords = [(h_quarter, 0, 3*h_quarter, W)]
            
        else:  # 'I': 四肢分块
            # 计算四肢区域的尺寸和位置
            w_upper = int(W * 0.2)  # 上肢宽度为20%
            w_lower = int(W * 0.3)  # 下肢宽度为30%
            h_upper = int(H * 0.3)  # 上肢高度为30%
            h_lower = int(H * 0.3)  # 下肢高度为30%
            y_upper = int(H * 0.2)  # 上肢起始位置为20%H
            y_lower = int(H * 0.7)  # 下肢起始位置为70%H
            
            coords = [
                (y_upper, 0, y_upper + h_upper, w_upper),                # 左上肢
                (y_upper, W - w_upper, y_upper + h_upper, W),            # 右上肢
                (y_lower, 0, y_lower + h_lower, w_lower),                # 左下肢
                (y_lower, W - w_lower, y_lower + h_lower, W)             # 右下肢
            ]
            
        return coords
        
    def forward(self, x):
        """自适应分块策略
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            patches: 分块后的特征图 [B, N, C, H', W']
            where N is number of patches
        """
        B, C, H, W = x.shape
        coords = self.layout_generator(H, W)
        
        patches = []
        for y1, x1, y2, x2 in coords:
            patch = x[:, :, y1:y2, x1:x2]
            patches.append(patch)
            
        # 堆叠所有patches
        patches = torch.stack(patches, dim=1)  # [B, N, C, H', W']
        return patches
    
    def merge_patches(self, patches, output_size):
        """将分块合并回原始尺寸，使用加权平均处理重叠区域
        Args:
            patches: 分块特征图列表 [B, N, C, H', W']
            output_size: 输出尺寸 (H, W)
        Returns:
            merged: 合并后的特征图 [B, C, H, W]
        """
        B = patches.shape[0]
        H, W = output_size
        coords = self.layout_generator(H, W)
        
        # 初始化输出特征图和权重图
        merged = torch.zeros(B, patches.shape[2], H, W, device=patches.device)
        weights = torch.zeros(B, 1, H, W, device=patches.device)
        
        # 将每个patch放回原位置并累加
        for i, (y1, x1, y2, x2) in enumerate(coords):
            patch = patches[:, i]
            merged[:, :, y1:y2, x1:x2] += patch
            weights[:, :, y1:y2, x1:x2] += 1.0
            
        # 使用权重进行归一化处理重叠区域
        weights = weights.clamp(min=1.0)  # 避免除零
        merged = merged / weights
        
        return merged