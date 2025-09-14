#!/usr/bin/env python3
"""
Marigold深度模型性能测试脚本
Marigold Depth Model Performance Testing Script

测试内容:
- 计算量 (FLOPs)
- 推理延迟 (Inference Latency) 
- 最大占用内存 (Maximum Memory Usage)
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict
import gc
import tracemalloc
from marigold.marigold_depth_pipeline import MarigoldDepthPipeline
from diffusers import AutoencoderTiny

def get_memory_usage() -> float:
    """获取当前内存使用量 (MB)"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def get_gpu_memory_usage() -> float:
    """获取GPU显存使用量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def count_parameters(model) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def create_test_image(height: int = 518, width: int = 518) -> torch.Tensor:
    """创建测试图像"""
    # 创建随机RGB图像 [1, 3, H, W]
    array = torch.randint(0, 256, (1, 3, height, width), dtype=torch.float16)
    return array

def test_inference_latency(pipeline, test_image: torch.Tensor, num_runs: int = 5) -> Dict[str, float]:
    """测试推理延迟"""
    latencies = []
    
    # 将测试图像移动到正确的设备和数据类型
    test_image = test_image.to(device=pipeline.device, dtype=pipeline.dtype)
    
    # 预热GPU
    _ = pipeline(test_image, denoising_steps=1, show_progress_bar=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    for i in range(num_runs):
        start_time = time.time()
        _ = pipeline(test_image, denoising_steps=10, show_progress_bar=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
        print(f"Run {i+1}: {latency:.3f}s")
    
    return {
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies)
    }

def test_memory_usage(pipeline, test_image: torch.Tensor) -> Dict[str, float]:
    """测试内存使用量"""
    # 将测试图像移动到正确的设备和数据类型
    test_image = test_image.to(device=pipeline.device, dtype=pipeline.dtype)
    
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 记录初始内存
    initial_ram = get_memory_usage()
    initial_gpu = get_gpu_memory_usage()
    
    # 开始内存追踪
    tracemalloc.start()
    
    # 执行推理
    _ = pipeline(test_image, denoising_steps=10, show_progress_bar=False)
    
    # 获取峰值内存
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # 记录推理后内存
    final_ram = get_memory_usage()
    final_gpu = get_gpu_memory_usage()
    
    return {
        'ram_initial_mb': initial_ram,
        'ram_final_mb': final_ram,
        'ram_peak_mb': peak_ram / 1024 / 1024,
        'ram_used_mb': final_ram - initial_ram,
        'gpu_initial_mb': initial_gpu,
        'gpu_final_mb': final_gpu,
        'gpu_used_mb': final_gpu - initial_gpu
    }

def main():
    """主测试函数"""
    print("=== Marigold深度模型性能测试 ===\n")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载Marigold深度模型...")
    try:
        pipeline = MarigoldDepthPipeline.from_pretrained(
            "checkpoint/marigold-depth-v1-1",
            torch_dtype=torch.float16,
            variant="fp16" if device == "cuda" else None
        ).to(device)
        # vae_tiny = AutoencoderTiny.from_pretrained("/home/daria/deeplearning/Marigold/checkpoint/taesd ", torch_dtype=torch.float16 )
        # pipeline.vae = vae_tiny.to(device)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建测试图像
    test_image = create_test_image(518, 518)
    print(f"测试图像尺寸: {test_image.shape}")
    
    # 1. 模型参数统计
    print("\n=== 1. 模型参数统计 ===")
    unet_params = count_parameters(pipeline.unet)
    vae_params = count_parameters(pipeline.vae)
    text_encoder_params = count_parameters(pipeline.text_encoder)
    total_params = unet_params + vae_params + text_encoder_params
    
    print(f"U-Net参数量: {unet_params:,}")
    print(f"VAE参数量: {vae_params:,}")
    print(f"文本编码器参数量: {text_encoder_params:,}")
    print(f"总参数量: {total_params:,}")
    

    
    # 3. 推理延迟测试
    print("\n=== 3. 推理延迟测试 ===")
    latency_results = test_inference_latency(pipeline, test_image, num_runs=5)
    print(f"平均延迟: {latency_results['mean_latency']:.3f}s")
    print(f"延迟标准差: {latency_results['std_latency']:.3f}s")
    print(f"最小延迟: {latency_results['min_latency']:.3f}s")
    print(f"最大延迟: {latency_results['max_latency']:.3f}s")
    
    # 4. 内存使用测试
    print("\n=== 4. 内存使用测试 ===")
    memory_results = test_memory_usage(pipeline, test_image)
    print(f"RAM初始: {memory_results['ram_initial_mb']:.1f} MB")
    print(f"RAM推理后: {memory_results['ram_final_mb']:.1f} MB")
    print(f"RAM峰值: {memory_results['ram_peak_mb']:.1f} MB")
    print(f"RAM增加: {memory_results['ram_used_mb']:.1f} MB")
    
    if torch.cuda.is_available():
        print(f"GPU显存初始: {memory_results['gpu_initial_mb']:.1f} MB")
        print(f"GPU显存推理后: {memory_results['gpu_final_mb']:.1f} MB")
        print(f"GPU显存增加: {memory_results['gpu_used_mb']:.1f} MB")
    
    # 5. 不同分辨率测试
    print("\n=== 5. 不同分辨率性能对比 ===")
    resolutions = [(256, 256), (512, 512), (768, 768)]
    
    for h, w in resolutions:
        print(f"\n测试分辨率: {w}x{h}")
        test_img = create_test_image(h, w)
        test_img = test_img.to(device=pipeline.device, dtype=pipeline.dtype)
        
        # 测试延迟
        start_time = time.time()
        _ = pipeline(test_img, denoising_steps=5, show_progress_bar=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"  延迟: {latency:.3f}s")
        
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()