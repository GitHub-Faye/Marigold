# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import sys
import os

# 将上层目录添加到Python路径中，用于导入marigold模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import os
import torch
from PIL import Image
from glob import glob
from tqdm.auto import tqdm

# 导入Marigold深度估计管道和输出类
from marigold import MarigoldDepthPipeline, MarigoldDepthOutput

# 支持的图像文件扩展名列表
EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    # 配置日志级别
    logging.basicConfig(level=logging.INFO)

    # -------------------- 参数解析 --------------------
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Marigold : 单目深度估计 : 多图像推理"
    )
    # 模型检查点路径或HuggingFace Hub名称
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-depth-v1-1",
        help="模型检查点路径或hub名称",
    )
    # 输入RGB图像文件夹路径
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="输入图像文件夹的路径",
    )
    # 输出目录路径
    parser.add_argument(
        "--output_dir", type=str, required=True, help="输出目录"
    )
    # 扩散模型去噪步数
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="扩散去噪步数，步数越多精度越高但推理速度越慢。如果设为None，将从检查点读取默认值",
    )
    # 处理分辨率
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="执行估计前输入图像调整到的分辨率。0表示使用原始输入分辨率；None表示从模型检查点解析最佳默认值。默认：None",
    )
    # 集成大小
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="要集成的预测数量。默认：1",
    )
    # 半精度模式
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="使用半精度(16位浮点)运行，可能导致次优结果",
    )
    # 输出处理分辨率
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="设置此标志将以processing_res的有效值输出结果，否则输出将调整为输入分辨率",
    )
    # 重采样方法
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="用于调整图像和预测大小的重采样方法。可以是bilinear、bicubic或nearest之一。默认：bilinear",
    )
    # 颜色映射
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="用于可视化深度预测的颜色映射",
    )
    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="可重现性种子。设为None表示随机推理。默认：None",
    )
    # 批处理大小
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="推理批处理大小。默认：0（将自动设置）",
    )
    # Apple Silicon支持
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="使用Apple Silicon进行更快推理（取决于可用性）",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 提取主要参数
    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    # 推理配置参数
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    # 检查集成大小是否过大
    if ensemble_size > 15:
        logging.warning("使用大集成大小运行会很慢")
    half_precision = args.half_precision

    # 分辨率和输出配置
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    # 检查分辨率配置的兼容性
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "在原生分辨率处理且不调整输出大小时，由于卷积层的填充和池化特性，可能无法得到完全相同的分辨率"
        )
    resample_method = args.resample_method

    # 其他配置参数
    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    # Apple Silicon默认批处理大小设置
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # 设置默认批处理大小

    # -------------------- 准备工作 --------------------
    # 创建输出目录
    output_dir_color = os.path.join(output_dir, "depth_colored")  # 彩色深度图目录
    output_dir_tif = os.path.join(output_dir, "depth_bw")        # 黑白深度图目录
    output_dir_npy = os.path.join(output_dir, "depth_npy")       # numpy数组目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"输出目录 = {output_dir}")

    # -------------------- 设备选择 --------------------
    # 根据可用性选择计算设备
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")  # 使用Apple Silicon MPS
        else:
            device = torch.device("cpu")
            logging.warning("MPS不可用。在CPU上运行会很慢")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")  # 使用NVIDIA CUDA
        else:
            device = torch.device("cpu")
            logging.warning("CUDA不可用。在CPU上运行会很慢")
    logging.info(f"设备 = {device}")

    # -------------------- 数据加载 --------------------
    # 获取所有图像文件
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    # 过滤出支持的图像格式
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    # 按文件名排序
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    # 检查是否找到图像文件
    if n_images > 0:
        logging.info(f"找到 {n_images} 张图像")
    else:
        logging.error(f"在 '{input_rgb_dir}' 中未找到图像")
        exit(1)

    # -------------------- 模型加载 --------------------
    # 设置数据类型和模型变体
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"使用半精度运行 ({dtype})，可能导致次优结果"
        )
    else:
        dtype = torch.float32
        variant = None

    # 从预训练检查点加载Marigold深度估计管道，使用AutoencoderTiny作为VAE
    from diffusers import AutoencoderTiny
    
    pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )
    
    # 替换VAE为madebyollin/taesd AutoencoderTiny
    vae_tiny = AutoencoderTiny.from_pretrained("/home/daria/deeplearning/Marigold/checkpoint/taesd ", torch_dtype=dtype)
    pipe.vae = vae_tiny
    logging.info("已将VAE替换为 madebyollin/taesd AutoencoderTiny")

    # 尝试启用xformers内存高效注意力机制
    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    # except ImportError:
    #     pass  # 如果没有xformers则跳过

    # 将模型移动到指定设备
    pipe = pipe.to(device)
    logging.info(
        f"已加载深度管道：scale_invariant={pipe.scale_invariant}, shift_invariant={pipe.shift_invariant}"
    )

    # 打印配置信息
    logging.info(
        f"推理设置：检查点 = `{checkpoint_path}`, "
        f"去噪步数 = {denoise_steps or pipe.default_denoising_steps}, "
        f"集成大小 = {ensemble_size}, "
        f"处理分辨率 = {processing_res or pipe.default_processing_resolution}, "
        f"种子 = {seed}; "
        f"颜色映射 = {color_map}"
    )

    # -------------------- 推理和保存 --------------------
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        # 遍历所有输入图像进行深度估计
        for rgb_path in tqdm(rgb_filename_list, desc="深度推理", leave=True):
            # 读取输入图像
            input_image = Image.open(rgb_path)

            # 设置随机数生成器
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # 执行深度估计推理
            pipe_out: MarigoldDepthOutput = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            # 提取推理结果
            depth_pred: np.ndarray = pipe_out.depth_np      # 深度预测数组
            depth_colored: Image.Image = pipe_out.depth_colored  # 彩色深度图

            # 保存为numpy数组格式
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_depth"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"现有文件：'{npy_save_path}' 将被覆盖")
            np.save(npy_save_path, depth_pred)

            # 保存为16位无符号整数PNG格式
            depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"现有文件：'{png_save_path}' 将被覆盖")
            Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            # 保存彩色深度图
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"现有文件：'{colored_save_path}' 将被覆盖"
                )
            depth_colored.save(colored_save_path)
