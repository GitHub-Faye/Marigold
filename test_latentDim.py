import torch
import logging
from diffusers import AutoencoderTiny

# 设置日志
logging.basicConfig(level=logging.INFO)

# 从本地Marigold模块导入管道
from marigold.marigold_depth_pipeline import MarigoldDepthPipeline

def test_vae_latent_shapes():
    """测试替换VAE前后的隐变量形状变化"""
    
    # 创建测试输入图像张量 (768x768)
    x = torch.randn(1, 3, 768, 768).to("cuda").to(torch.float16)
    print(f"输入图像形状: {x.shape}")
    
    # 从预训练检查点加载Marigold深度估计管道
    print("正在加载Marigold深度估计管道...")
    pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
        "/home/daria/deeplearning/Marigold/checkpoint/marigold-depth-v1-1", 
        variant='fp16', 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    print(f"原始VAE类型: {type(pipe.vae).__name__}")
    print(f"原始latent_scale_factor: {pipe.latent_scale_factor}")
    
    # 测试原始VAE的编码
    with torch.no_grad():
        rgb_latent_original = pipe.encode_rgb(x)
        print(f"原始VAE潜在变量形状: {rgb_latent_original.shape}")
        
        # 测试解码
        depth_decoded_original = pipe.decode_depth(rgb_latent_original)
        print(f"原始VAE解码深度形状: {depth_decoded_original.shape}")
        print(f"原始VAE潜在变量范围: [{rgb_latent_original.min():.4f}, {rgb_latent_original.max():.4f}]")
        print(f"原始VAE解码深度范围: [{depth_decoded_original.min():.4f}, {depth_decoded_original.max():.4f}]")
    
    # 替换VAE为AutoencoderTiny
    print("\n正在替换VAE为AutoencoderTiny...")
    try:
        vae_tiny = AutoencoderTiny.from_pretrained(
            "/home/daria/deeplearning/Marigold/checkpoint/taesd ", 
            torch_dtype=torch.float16
        )
        vae_tiny = vae_tiny.to("cuda")
        
        # 保存原始VAE以便比较
        original_vae = pipe.vae
        original_scale_factor = pipe.latent_scale_factor
        
        # 替换VAE
        pipe.vae = vae_tiny
        pipe.latent_scale_factor = vae_tiny.scaling_factor
        
        print(f"新VAE类型: {type(pipe.vae).__name__}")
        print(f"新latent_scale_factor: {pipe.latent_scale_factor}")
        
        # 测试新VAE的编码
        with torch.no_grad():
            rgb_latent_tiny = pipe.encode_rgb(x)
            print(f"AutoencoderTiny潜在变量形状: {rgb_latent_tiny.shape}")
            
            # 测试解码
            depth_decoded_tiny = pipe.decode_depth(rgb_latent_tiny)
            print(f"AutoencoderTiny解码深度形状: {depth_decoded_tiny.shape}")
            print(f"AutoencoderTiny潜在变量范围: [{rgb_latent_tiny.min():.4f}, {rgb_latent_tiny.max():.4f}]")
            print(f"AutoencoderTiny解码深度范围: [{depth_decoded_tiny.min():.4f}, {depth_decoded_tiny.max():.4f}]")
        
        # 比较形状变化和数值范围
        print(f"\n形状变化总结:")
        print(f"输入图像: {x.shape}")
        print(f"原始VAE潜在变量: {rgb_latent_original.shape} (压缩比: 1/{768//rgb_latent_original.shape[-1]})")
        print(f"AutoencoderTiny潜在变量: {rgb_latent_tiny.shape} (压缩比: 1/{768//rgb_latent_tiny.shape[-1]})")
        print(f"原始VAE解码输出: {depth_decoded_original.shape}")
        print(f"AutoencoderTiny解码输出: {depth_decoded_tiny.shape}")
        
        print(f"\n数值范围比较:")
        print(f"原始VAE潜在变量范围: [{rgb_latent_original.min():.4f}, {rgb_latent_original.max():.4f}]")
        print(f"AutoencoderTiny潜在变量范围: [{rgb_latent_tiny.min():.4f}, {rgb_latent_tiny.max():.4f}]")
        print(f"原始VAE解码深度范围: [{depth_decoded_original.min():.4f}, {depth_decoded_original.max():.4f}]")
        print(f"AutoencoderTiny解码深度范围: [{depth_decoded_tiny.min():.4f}, {depth_decoded_tiny.max():.4f}]")
        
        # 恢复原始VAE
        pipe.vae = original_vae
        pipe.latent_scale_factor = original_scale_factor
        
        logging.info("VAE替换测试完成，已恢复原始VAE")
        
    except Exception as e:
        print(f"替换VAE时出错: {e}")
        print("请检查AutoencoderTiny模型路径是否正确")

if __name__ == "__main__":
    test_vae_latent_shapes()