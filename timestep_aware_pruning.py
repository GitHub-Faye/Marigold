import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from .importance import Importance
from diffusers import DDIMScheduler

class TimestepAwareImportance(Importance):
    """
    基于您观察的时间步感知重要度评估器
    
    核心思想：
    1. 不同去噪步数下，参数对中间步骤误差的影响权重不同
    2. 高步数推理中，早期步骤的误差会累积放大
    3. 对深度估计，结构信息比细节更重要
    """
    
    def __init__(self, 
                 scheduler: DDIMScheduler,
                 step_configs: List[int] = [1, 2, 4, 8, 10],
                 error_weight_strategy: str = "cumulative",
                 importance_fusion: str = "weighted_sum",
                 depth_structure_bias: float = 2.0):
        """
        Args:
            scheduler: DDIM调度器，用于计算时间步分布
            step_configs: 要分析的不同去噪步数配置
            error_weight_strategy: 误差权重策略 ("cumulative", "exponential", "linear")
            importance_fusion: 重要度融合方式 ("weighted_sum", "max", "harmonic_mean")
            depth_structure_bias: 深度结构偏置，强调结构相关参数
        """
        self.scheduler = scheduler
        self.step_configs = step_configs
        self.error_weight_strategy = error_weight_strategy
        self.importance_fusion = importance_fusion
        self.depth_structure_bias = depth_structure_bias
        
        # 预计算不同步数配置的时间步权重
        self.timestep_weights = self._compute_timestep_weights()
    
    def _compute_timestep_weights(self) -> Dict[int, torch.Tensor]:
        """计算不同步数配置下的时间步权重"""
        weights = {}
        
        for num_steps in self.step_configs:
            self.scheduler.set_timesteps(num_steps)
            timesteps = self.scheduler.timesteps
            
            # 计算每个时间步的噪声水平
            noise_levels = []
            for t in timesteps:
                alpha_prod = self.scheduler.alphas_cumprod[t]
                noise_level = (1 - alpha_prod).sqrt()
                noise_levels.append(noise_level.item())
            
            # 根据策略计算权重
            step_weights = self._calculate_step_weights(noise_levels, num_steps)
            weights[num_steps] = torch.tensor(step_weights, dtype=torch.float32)
            
        return weights
    
    def _calculate_step_weights(self, noise_levels: List[float], num_steps: int) -> List[float]:
        """根据噪声水平和策略计算步骤权重"""
        if self.error_weight_strategy == "cumulative":
            # 累积误差策略：早期步骤权重更高，因为误差会传播
            weights = []
            for i, noise in enumerate(noise_levels):
                # 剩余步数越多，权重越高
                remaining_steps = num_steps - i
                cumulative_weight = remaining_steps * (1.0 - noise)
                weights.append(cumulative_weight)
                
        elif self.error_weight_strategy == "exponential":
            # 指数衰减：噪声水平高的步骤权重指数下降
            weights = [np.exp(-noise * 3.0) for noise in noise_levels]
            
        elif self.error_weight_strategy == "linear":
            # 线性策略：简单的线性权重分配
            weights = [1.0 - noise for noise in noise_levels]
            
        else:
            weights = [1.0] * len(noise_levels)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            
        return weights
    
    def _compute_parameter_structural_importance(self, layer, param_indices) -> torch.Tensor:
        """
        计算参数的结构重要度
        对于深度估计，结构信息比细节更重要
        """
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            # 卷积层：考虑感受野和特征图结构
            weight = layer.weight.data
            
            if len(param_indices) > 0:
                selected_weights = weight[param_indices] if weight.dim() > 2 else weight
                
                # 结构重要度：更大的卷积核和更多连接的通道更重要
                spatial_importance = selected_weights.abs().sum(dim=(2, 3)) if weight.dim() == 4 else selected_weights.abs().sum(dim=1)
                channel_connectivity = selected_weights.abs().sum(dim=1).sum(dim=(1, 2)) if weight.dim() == 4 else selected_weights.abs().sum(dim=0)
                
                structural_score = spatial_importance.mean(dim=1) + channel_connectivity * 0.5
                return structural_score * self.depth_structure_bias
            
        elif isinstance(layer, nn.Linear):
            # 线性层：考虑连接强度
            weight = layer.weight.data
            if len(param_indices) > 0:
                selected_weights = weight[param_indices]
                connection_strength = selected_weights.abs().sum(dim=1)
                return connection_strength * self.depth_structure_bias
        
        # 默认情况
        return torch.ones(len(param_indices))
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1) -> torch.Tensor:
        """
        计算考虑时间步的参数重要度
        """
        multi_step_importances = []
        
        # 对每个步数配置计算重要度
        for num_steps in self.step_configs:
            step_importance = self._compute_single_config_importance(
                group, num_steps, ch_groups
            )
            if step_importance is not None:
                # 应用时间步权重
                step_weights = self.timestep_weights[num_steps]
                weighted_importance = step_importance * step_weights.mean()
                multi_step_importances.append(weighted_importance)
        
        if len(multi_step_importances) == 0:
            return None
            
        # 融合多个配置的重要度
        return self._fuse_importances(multi_step_importances)
    
    def _compute_single_config_importance(self, group, num_steps: int, ch_groups: int) -> Optional[torch.Tensor]:
        """计算单个步数配置下的重要度"""
        group_imp = []
        
        for dep, idxs in group:
            layer = dep.target.module
            prune_fn = dep.handler
            
            # 基础梯度重要度（如果有梯度）
            if hasattr(layer.weight, 'grad') and layer.weight.grad is not None:
                base_importance = self._compute_gradient_importance(layer, idxs, prune_fn)
            else:
                base_importance = self._compute_magnitude_importance(layer, idxs, prune_fn)
            
            if base_importance is not None:
                # 结合结构重要度
                structural_importance = self._compute_parameter_structural_importance(layer, idxs)
                
                # 融合基础重要度和结构重要度
                if len(base_importance) == len(structural_importance):
                    combined_importance = base_importance + structural_importance
                else:
                    combined_importance = base_importance
                
                group_imp.append(combined_importance)
        
        if len(group_imp) == 0:
            return None
            
        # 对齐和融合组内重要度
        return self._align_and_reduce_group_importance(group_imp)
    
    def _compute_gradient_importance(self, layer, idxs, prune_fn) -> Optional[torch.Tensor]:
        """基于梯度计算重要度"""
        if prune_fn.__name__.endswith('_out_channels'):
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                dw = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
            else:
                w = layer.weight.data[idxs].flatten(1)
                dw = layer.weight.grad.data[idxs].flatten(1)
            
            # Taylor expansion importance
            taylor_importance = (w * dw).abs().sum(1)
            return taylor_importance
        
        elif prune_fn.__name__.endswith('_in_channels'):
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.flatten(1)[idxs]
                dw = layer.weight.grad.flatten(1)[idxs]
            else:
                w = layer.weight.transpose(0, 1).flatten(1)[idxs]
                dw = layer.weight.grad.transpose(0, 1).flatten(1)[idxs]
            
            taylor_importance = (w * dw).abs().sum(1)
            return taylor_importance
        
        return None
    
    def _compute_magnitude_importance(self, layer, idxs, prune_fn) -> Optional[torch.Tensor]:
        """基于权重幅度计算重要度"""
        if prune_fn.__name__.endswith('_out_channels'):
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
            else:
                w = layer.weight.data[idxs].flatten(1)
            
            magnitude_importance = w.abs().pow(2).sum(1)
            return magnitude_importance
        
        elif prune_fn.__name__.endswith('_in_channels'):
            if hasattr(layer, "transposed") and layer.transposed:
                w = layer.weight.flatten(1)[idxs]
            else:
                w = layer.weight.transpose(0, 1).flatten(1)[idxs]
            
            magnitude_importance = w.abs().pow(2).sum(1)
            return magnitude_importance
        
        return None
    
    def _align_and_reduce_group_importance(self, group_imp: List[torch.Tensor]) -> torch.Tensor:
        """对齐和归约组内重要度"""
        if len(group_imp) == 1:
            return group_imp[0]
        
        # 找到统一的尺寸
        target_size = len(group_imp[0])
        aligned_imp = []
        
        for imp in group_imp:
            if len(imp) == target_size:
                aligned_imp.append(imp)
        
        if len(aligned_imp) == 0:
            return group_imp[0]
        
        # 叠加并归约
        stacked_imp = torch.stack(aligned_imp, dim=0)
        return stacked_imp.mean(dim=0)  # 可以改为sum或其他策略
    
    def _fuse_importances(self, importances: List[torch.Tensor]) -> torch.Tensor:
        """融合多个步数配置的重要度"""
        if len(importances) == 1:
            return importances[0]
        
        stacked = torch.stack(importances, dim=0)
        
        if self.importance_fusion == "weighted_sum":
            # 给更高步数更大权重（因为误差累积更重要）
            weights = torch.tensor([float(i+1) for i in range(len(importances))], 
                                 dtype=stacked.dtype)
            weights = weights / weights.sum()
            return (stacked * weights.view(-1, 1)).sum(dim=0)
        
        elif self.importance_fusion == "max":
            return stacked.max(dim=0)[0]
        
        elif self.importance_fusion == "harmonic_mean":
            # 调和平均，更保守的融合策略
            reciprocal_mean = (1.0 / (stacked + 1e-8)).mean(dim=0)
            return 1.0 / reciprocal_mean
        
        else:  # default: simple mean
            return stacked.mean(dim=0)


class DepthEstimationPruner:
    """
    针对深度估计任务的专用剪枝器
    结合时间步感知重要度和深度估计特性
    """
    
    def __init__(self, 
                 model: nn.Module,
                 importance_evaluator: TimestepAwareImportance,
                 preserve_structure_layers: List[str] = None):
        self.model = model
        self.importance_evaluator = importance_evaluator
        self.preserve_structure_layers = preserve_structure_layers or []
    
    def analyze_layer_sensitivity(self) -> Dict[str, float]:
        """
        分析各层对不同去噪步数的敏感度
        """
        layer_sensitivity = {}
        
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # 计算该层在不同步数配置下的重要度方差
                step_importances = []
                for num_steps in self.importance_evaluator.step_configs:
                    # 模拟计算该层重要度...
                    pass
                
                if len(step_importances) > 1:
                    importance_std = np.std(step_importances)
                    layer_sensitivity[name] = importance_std
        
        return layer_sensitivity
    
    def recommend_pruning_strategy(self) -> Dict[str, any]:
        """
        基于分析结果推荐剪枝策略
        """
        sensitivity = self.analyze_layer_sensitivity()
        
        strategy = {
            "high_sensitivity_layers": [],
            "low_sensitivity_layers": [],
            "recommended_pruning_ratios": {},
            "preserve_layers": self.preserve_structure_layers
        }
        
        if sensitivity:
            sensitivity_values = list(sensitivity.values())
            threshold = np.percentile(sensitivity_values, 70)  # 70分位数作为阈值
            
            for layer_name, sens in sensitivity.items():
                if sens > threshold:
                    strategy["high_sensitivity_layers"].append(layer_name)
                    strategy["recommended_pruning_ratios"][layer_name] = 0.1  # 保守剪枝
                else:
                    strategy["low_sensitivity_layers"].append(layer_name)
                    strategy["recommended_pruning_ratios"][layer_name] = 0.3  # 积极剪枝
        
        return strategy