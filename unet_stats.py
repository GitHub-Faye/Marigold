#!/usr/bin/env python3
"""
UNET网络统计工具
统计传入UNET网络的参数量、MAC数以及在768x768输入图像上的推理速度
"""

import time
import torch
from typing import Union, Dict
import argparse
from contextlib import contextmanager

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop library not found. MAC calculation will be skipped.")
    print("Install with: pip install thop")

try:
    from fvcore.nn import FlopCountMode, flop_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

from diffusers import UNet2DConditionModel
from marigold.marigold_depth_pipeline import MarigoldDepthPipeline


@contextmanager
def torch_timing():
    """PyTorch推理时间测量上下文管理器"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    return end_time - start_time


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def calculate_macs_thop(model: torch.nn.Module, input_tensors: tuple) -> int:
    """使用thop计算MAC数"""
    if not HAS_THOP:
        return None
    
    try:
        # 确保模型和输入在相同设备上且为float32
        device = next(model.parameters()).device
        model_copy = model.eval().to(device)
        
        # 转换输入为float32并移到正确设备
        input_tensors_float32 = tuple(
            t.to(device).float() if t.dtype == torch.float16 else t.to(device) 
            for t in input_tensors
        )
        
        # 临时转换模型为float32
        original_dtype = next(model.parameters()).dtype
        if original_dtype == torch.float16:
            model_copy = model_copy.float()
        
        macs, _ = profile(model_copy, inputs=input_tensors_float32, verbose=False)
        return macs
    except Exception as e:
        print(f"THOP MAC calculation failed: {e}")
        return None


def calculate_macs_fvcore(model: torch.nn.Module, input_tensors: tuple) -> int:
    """使用fvcore计算FLOP数"""
    if not HAS_FVCORE:
        return None
    
    try:
        model_copy = model.eval()
        with torch.no_grad():
            flop_dict, _ = flop_count(model_copy, input_tensors, mode=FlopCountMode.FlopCount)
            total_flops = sum(flop_dict.values())
        return total_flops // 2  # FLOP转MAC (大约)
    except Exception as e:
        print(f"FVCore FLOP calculation failed: {e}")
        return None


def measure_inference_speed(model: torch.nn.Module, input_tensors: tuple, 
                          device: str = 'cuda', num_runs: int = 10, warmup_runs: int = 3) -> Dict[str, float]:
    """测量推理速度"""
    model = model.to(device).eval()
    
    # 确保输入张量与模型在相同设备和数据类型
    model_dtype = next(model.parameters()).dtype
    input_tensors = tuple(
        t.to(device).to(model_dtype) if t.dtype != model_dtype else t.to(device) 
        for t in input_tensors
    )
    
    # 预热运行
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*input_tensors)
    
    # 实际测量
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(*input_tensors)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': torch.tensor(times).std().item()
    }


def create_unet_inputs(unet_model: UNet2DConditionModel, device: str = 'cuda', 
                      height: int = 768, width: int = 768) -> tuple:
    """为UNET创建输入张量"""
    # 获取UNET配置
    in_channels = unet_model.config.in_channels
    
    # 计算潜在空间尺寸 (通常是原图的1/8)
    latent_height = height // 8
    latent_width = width // 8
    
    # 创建输入张量
    latents = torch.randn(1, in_channels, latent_height, latent_width, device=device)
    timestep = torch.tensor([1], device=device)
    encoder_hidden_states = torch.randn(1, 77, 1024, device=device)  # CLIP text embedding
    
    return (latents, timestep, encoder_hidden_states)


def analyze_unet_from_pipeline(pipeline_path: str, device: str = 'cuda') -> Dict:
    """从Marigold pipeline中分析UNET"""
    print(f"Loading Marigold pipeline from: {pipeline_path}")
    
    try:
        # 加载pipeline
        pipeline = MarigoldDepthPipeline.from_pretrained(
            pipeline_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            variant="fp16" if device == 'cuda' else None
        )
        
        unet = pipeline.unet
        print(f"Loaded UNET: {unet.__class__.__name__}")
        print(f"UNET config: in_channels={unet.config.in_channels}, "
              f"out_channels={unet.config.out_channels}, "
              f"sample_size={unet.config.sample_size}")
        
        return analyze_unet_model(unet, device)
        
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return {}


def analyze_unet_model(unet: UNet2DConditionModel, device: str = 'cuda', 
                      height: int = 768, width: int = 768) -> Dict:
    """分析UNET模型"""
    print(f"\n{'='*60}")
    print(f"UNET Model Analysis (Input: {height}x{width})")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. UNET网络结构
    print("\n1. UNET Network Architecture:")
    print(f"   Model Class: {unet.__class__.__name__}")
    print(f"   Input Channels: {unet.config.in_channels}")
    print(f"   Output Channels: {unet.config.out_channels}")
    print(f"   Sample Size: {unet.config.sample_size}")
    print(f"   Down Block Types: {unet.config.down_block_types}")
    print(f"   Up Block Types: {unet.config.up_block_types}")
    print(f"   Block Out Channels: {unet.config.block_out_channels}")
    print(f"   Layers Per Block: {unet.config.layers_per_block}")
    print(f"   Attention Head Dim: {unet.config.attention_head_dim}")
    print(f"   Cross Attention Dim: {unet.config.cross_attention_dim}")
    print(f"   Use Linear Projection: {unet.config.use_linear_projection}")
    print(f"   Class Embed Type: {unet.config.class_embed_type}")
    print(f"   Time Embedding Type: {unet.config.time_embedding_type}")
    
    # 保存网络结构信息
    results['architecture'] = {
        'model_class': unet.__class__.__name__,
        'in_channels': unet.config.in_channels,
        'out_channels': unet.config.out_channels,
        'sample_size': unet.config.sample_size,
        'down_block_types': unet.config.down_block_types,
        'up_block_types': unet.config.up_block_types,
        'block_out_channels': unet.config.block_out_channels,
        'layers_per_block': unet.config.layers_per_block,
        'attention_head_dim': unet.config.attention_head_dim,
        'cross_attention_dim': unet.config.cross_attention_dim,
    }
    
    # 2. 详细网络结构
    print(f"\n2. Detailed Network Structure:")
    print(unet)
    
    # 3. 参数统计
    print(f"\n3. Parameter Statistics:")
    param_stats = count_parameters(unet)
    results['parameters'] = param_stats
    
    total_params_m = param_stats['total_params'] / 1e6
    trainable_params_m = param_stats['trainable_params'] / 1e6
    
    print(f"   Total Parameters: {param_stats['total_params']:,} ({total_params_m:.2f}M)")
    print(f"   Trainable Parameters: {param_stats['trainable_params']:,} ({trainable_params_m:.2f}M)")
    print(f"   Non-trainable Parameters: {param_stats['non_trainable_params']:,}")
    
    # 4. 创建输入
    input_tensors = create_unet_inputs(unet, device, height, width)
    
    print(f"\n4. Input Tensors:")
    print(f"   Latents shape: {input_tensors[0].shape}")
    print(f"   Timestep shape: {input_tensors[1].shape}")
    print(f"   Text embeddings shape: {input_tensors[2].shape}")
    
    # 5. MAC计算
    print("\n5. MAC Calculation:")
    
    # 使用thop
    if HAS_THOP:
        macs_thop = calculate_macs_thop(unet, input_tensors)
        if macs_thop is not None:
            macs_thop_g = macs_thop / 1e9
            print(f"   MACs (thop): {macs_thop:,} ({macs_thop_g:.2f}G)")
            results['macs_thop'] = macs_thop
        else:
            print("   MACs (thop): Failed to calculate")
    
    # 使用fvcore
    if HAS_FVCORE:
        macs_fvcore = calculate_macs_fvcore(unet, input_tensors)
        if macs_fvcore is not None:
            macs_fvcore_g = macs_fvcore / 1e9
            print(f"   MACs (fvcore): {macs_fvcore:,} ({macs_fvcore_g:.2f}G)")
            results['macs_fvcore'] = macs_fvcore
        else:
            print("   MACs (fvcore): Failed to calculate")
    
    if not HAS_THOP and not HAS_FVCORE:
        print("   No MAC calculation libraries available")
    
    # 6. 推理速度测试
    print("\n6. Inference Speed Test:")
    try:
        speed_stats = measure_inference_speed(unet, input_tensors, device)
        results['inference_speed'] = speed_stats
        
        print(f"   Average time: {speed_stats['avg_time']*1000:.2f} ms")
        print(f"   Min time: {speed_stats['min_time']*1000:.2f} ms")
        print(f"   Max time: {speed_stats['max_time']*1000:.2f} ms")
        print(f"   Std deviation: {speed_stats['std_time']*1000:.2f} ms")
        
        fps = 1.0 / speed_stats['avg_time']
        print(f"   Approximate FPS: {fps:.2f}")
        
    except Exception as e:
        print(f"   Inference speed test failed: {e}")
    
    # 7. 内存使用情况
    if device == 'cuda' and torch.cuda.is_available():
        print("\n7. GPU Memory Usage:")
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   Allocated: {memory_allocated:.2f} GB")
        print(f"   Reserved: {memory_reserved:.2f} GB")
        results['gpu_memory'] = {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved
        }
    
    print(f"\n{'='*60}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze UNET model statistics')
    parser.add_argument('--model_path', type=str, 
                       help='Path to Marigold pipeline or UNET model')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--height', type=int, default=768, 
                       help='Input image height')
    parser.add_argument('--width', type=int, default=768, 
                       help='Input image width')
    parser.add_argument('--num_runs', type=int, default=10, 
                       help='Number of inference runs for speed test')
    
    args = parser.parse_args()
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    if args.model_path:
        # 从指定路径加载模型
        results = analyze_unet_from_pipeline(args.model_path, args.device)
    else:
        # 使用默认的预训练模型
        print("No model path specified, using default Marigold model")
        default_model = "prs-eth/marigold-depth-v1-1"
        results = analyze_unet_from_pipeline(default_model, args.device)
    
    return results


if __name__ == "__main__":
    main()