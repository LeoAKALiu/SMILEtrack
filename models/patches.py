import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePatches(nn.Module):
    """自适应图像分块模块
    
    该模块实现了一个灵活的图像分块策略，可以将输入特征图划分为重叠或不重叠的小块。
    主要用于目标跟踪中的局部特征提取和相似度计算。
    
    参数:
        patch_size (int): 分块的大小，默认为16
        stride (int): 滑动窗口的步长，默认为8。当stride < patch_size时会产生重叠的块
    
    使用示例:
        >>> patches_module = AdaptivePatches(patch_size=16, stride=8)
        >>> x = torch.randn(1, 64, 128, 128)  # 输入特征图
        >>> patches = patches_module(x)  # 获取分块结果
        >>> merged = patches_module.merge_patches(patches, (128, 128))  # 合并回原始尺寸
    """
    def __init__(self, patch_size=16, stride=8):
        super(AdaptivePatches, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        
    def forward(self, x):
        """自适应分块策略
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            patches: 分块后的特征图 [B, N, C, P, P]
            where N is number of patches, P is patch_size
        """
        B, C, H, W = x.shape
        
        # 使用滑动窗口进行分块
        patches = F.unfold(x, 
                          kernel_size=(self.patch_size, self.patch_size),
                          stride=self.stride)
        
        # 重塑张量形状
        patches = patches.reshape(B, C, self.patch_size, self.patch_size, -1)
        patches = patches.permute(0, 4, 1, 2, 3)  # [B, N, C, P, P]
        
        return patches
    
    def merge_patches(self, patches, output_size):
        """将分块合并回原始尺寸
        Args:
            patches: 分块特征图 [B, N, C, P, P]
            output_size: 输出尺寸 (H, W)
        Returns:
            merged: 合并后的特征图 [B, C, H, W]
        """
        B, N, C, P, P = patches.shape
        
        # 重塑张量用于fold操作
        patches = patches.permute(0, 2, 3, 4, 1)  # [B, C, P, P, N]
        patches = patches.reshape(B, C * P * P, N)
        
        # 使用fold操作合并patches
        merged = F.fold(patches,
                       output_size=output_size,
                       kernel_size=(self.patch_size, self.patch_size),
                       stride=self.stride)
        
        return merged