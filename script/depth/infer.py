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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import os
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from src.util.seeding import seed_all


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- 参数设置 / Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : 单目深度估计 : 数据集推理 / Marigold : Monocular Depth Estimation : Dataset Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-depth-v1-1",
        help="检查点路径或hub名称 / Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="评估数据集配置文件的路径 / Path to the config file of the evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="数据集的基础路径 / Base path to the datasets.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="输出目录 / Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        required=True,
        help="扩散去噪步数 / Diffusion denoising steps.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        required=True,
        help="执行估计前输入图像调整的分辨率。`0`使用原始输入分辨率 / Resolution to which the input is resized before performing estimation. `0` uses the original input resolution.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        required=True,
        help="要集成的预测数量 / Number of predictions to be ensembled.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="使用半精度(16位浮点)运行，可能导致次优结果 / Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="设置此标志将以`processing_res`的有效值输出结果，否则输出将调整为输入分辨率 / Setting this flag will output the result at the effective value of `processing_res`, otherwise the output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="用于调整图像和预测大小的重采样方法。可以是`bilinear`、`bicubic`或`nearest`之一。默认：`bilinear` / Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="可重现性种子。设置为`None`进行随机推理。默认：`None` / Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("使用大的集成大小运行会很慢 / Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "在原生分辨率下处理而不调整输出大小可能不会得到完全相同的分辨率，这是由于卷积层的填充和池化特性 / Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    logging.debug(f"Arguments: {args}")

    # -------------------- 准备工作 / Preparation --------------------
    # 打印配置信息 / Print out config
    logging.info(
        f"推理设置：检查点 = `{checkpoint_path}`，去噪步数 = {denoise_steps}，集成大小 = {ensemble_size}，处理分辨率 = {processing_res}，种子 = {seed}；数据集配置 = `{dataset_config}` / Inference settings: checkpoint = `{checkpoint_path}`, with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, processing resolution = {processing_res}, seed = {seed}; dataset config = `{dataset_config}`."
    )

    # 随机种子 / Random seed
    if seed is None:
        import time

        seed = int(time.time())

    seed_all(seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(
                    f"目录'{directory}'已存在。确定要继续吗？(y/n) / The directory '{directory}' already exists. Are you sure to continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("退出中... / Exiting...")
                exit()
            else:
                print("无效输入。请输入'y'(是)或'n'(否) / Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # 递归调用再次询问 / Recursive call to ask again

    check_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"输出目录 = {output_dir} / output dir = {output_dir}")

    # -------------------- 设备 / Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA不可用。在CPU上运行会很慢 / CUDA is not available. Running on CPU will be slow.")
    logging.info(f"设备 = {device} / device = {device}")

    # -------------------- 数据 / Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    )
    assert isinstance(dataset, BaseDepthDataset)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- 模型 / Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"使用半精度({dtype})运行，可能导致次优结果 / Running with half precision ({dtype}), might lead to suboptimal result."
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
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("在没有xformers的情况下继续 / Proceeding without xformers")

    pipe = pipe.to(device)
    logging.info(
        f"已加载深度管道：scale_invariant={pipe.scale_invariant}, shift_invariant={pipe.shift_invariant} / Loaded depth pipeline: scale_invariant={pipe.scale_invariant}, shift_invariant={pipe.shift_invariant}"
    )

    # -------------------- 推理和保存 / Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"在{dataset.disp_name}上进行深度推理 / Depth Inference on {dataset.disp_name}", leave=True
        ):
            # 读取输入图像 / Read input image
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            input_image = Image.fromarray(rgb_int)

            # 随机数生成器 / Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # 执行推理 / Perform inference
            pipe_out: MarigoldDepthOutput = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=0,
                color_map=None,
                show_progress_bar=False,
                resample_method=resample_method,
                generator=generator,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            # 保存预测结果 / Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(output_dir, os.path.dirname(rgb_filename))
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            pred_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix=".npy"
            )
            save_to = os.path.join(scene_dir, pred_basename)
            if os.path.exists(save_to):
                logging.warning(f"现有文件:'{save_to}'将被覆盖 / Existing file: '{save_to}' will be overwritten")

            np.save(save_to, depth_pred)
