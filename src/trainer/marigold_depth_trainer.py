# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
# 版权所有 2023-2025 Marigold团队，苏黎世联邦理工学院。保留所有权利。
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

"""
Marigold深度估计训练器
Marigold Depth Estimation Trainer

这是Marigold深度估计模型的核心训练器类，负责：
1. 模型组件的初始化和配置（U-Net、VAE、文本编码器）
2. 训练循环的管理和优化器的更新
3. 扩散过程的前向传播和损失计算
4. 验证和可视化的执行
5. 检查点的保存和恢复

Key features:
- 仅训练U-Net，冻结VAE和文本编码器
- 支持梯度累积和多分辨率噪声
- 集成验证、可视化和检查点管理
- 支持断点续训和时间限制训练
"""

import logging
import numpy as np
import os
import shutil
import torch
from PIL import Image
from datetime import datetime
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Union

from marigold.marigold_depth_pipeline import MarigoldDepthPipeline, MarigoldDepthOutput
from src.util import metric
from src.util.alignment import align_depth_least_square
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence


class MarigoldDepthTrainer:
    """
    Marigold深度估计模型训练器
    
    管理整个训练流程，包括：
    - 模型组件的初始化和配置
    - 训练循环和优化过程
    - 验证和可视化
    - 检查点管理
    """
    
    def __init__(
        self,
        cfg: OmegaConf,                    # 配置文件 / Configuration
        model: MarigoldDepthPipeline,      # Marigold深度估计管道 / Marigold depth pipeline
        train_dataloader: DataLoader,     # 训练数据加载器 / Training dataloader
        device,                           # 计算设备 / Compute device
        out_dir_ckpt,                     # 检查点输出目录 / Checkpoint output directory
        out_dir_eval,                     # 评估结果输出目录 / Evaluation output directory
        out_dir_vis,                      # 可视化输出目录 / Visualization output directory
        accumulation_steps: int,          # 梯度累积步数 / Gradient accumulation steps
        val_dataloaders: List[DataLoader] = None,    # 验证数据加载器列表 / Validation dataloaders
        vis_dataloaders: List[DataLoader] = None,    # 可视化数据加载器列表 / Visualization dataloaders
    ):
        # 基础配置 / Basic configuration
        self.cfg: OmegaConf = cfg
        self.model: MarigoldDepthPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed  # 用于生成种子序列，设置为None则不使用种子训练 / Used to generate seed sequence, set to None to train w/o seeding
        )
        
        # 输出目录配置 / Output directory configuration
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        
        # 数据加载器配置 / Dataloader configuration
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # 适配输入层：修改U-Net输入通道从4到8 / Adapt input layers: modify U-Net input channels from 4 to 8
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        # 编码空文本提示 / Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        # 启用XFormers内存高效注意力 / Enable XFormers memory efficient attention
        self.model.unet.enable_xformers_memory_efficient_attention()

        # 设置可训练性：仅训练U-Net / Set trainability: only train U-Net
        self.model.vae.requires_grad_(False)            # 冻结VAE / Freeze VAE
        self.model.text_encoder.requires_grad_(False)   # 冻结文本编码器 / Freeze text encoder
        self.model.unet.requires_grad_(True)            # U-Net可训练 / U-Net trainable

        # 优化器配置（必须在输入层适配后定义） / Optimizer configuration (should be defined after input layer is adapted)
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # 学习率调度器配置 / Learning rate scheduler configuration
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,  # 总迭代次数 / Total iterations
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,       # 最终学习率比例 / Final LR ratio
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,     # 预热步数 / Warmup steps
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # 损失函数配置 / Loss function configuration
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # 训练噪声调度器配置 / Training noise scheduler configuration
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,    # 重新缩放beta为零信噪比 / Rescale betas to zero SNR
            timestep_spacing="trailing",    # 时间步间距：尾部间距 / Timestep spacing: trailing
        )

        logging.info(
            "DDPM训练噪声调度器配置已更新 / DDPM training noise scheduler config is updated: "
            f"rescale_betas_zero_snr = {self.training_noise_scheduler.config.rescale_betas_zero_snr}, "
            f"timestep_spacing = {self.training_noise_scheduler.config.timestep_spacing}"
        )

        # 预测类型配置 / Prediction type configuration
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "预测类型不同 / Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps  # 训练时间步数 / Number of training timesteps
        )

        # 推理DDIM调度器（用于验证） / Inference DDIM scheduler (used for validation)
        self.model.scheduler = DDIMScheduler.from_config(
            self.training_noise_scheduler.config,
        )

        # 评估指标配置 / Evaluation metrics configuration
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        # 指标追踪器初始化 / Metric tracker initialization
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])

        # 主要验证指标（用于最佳检查点保存） / Main validation metric (for best checkpoint saving)
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal

        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"主要评估指标 `{self.main_val_metric}` 未在评估指标中找到 / Main eval metric `{self.main_val_metric}` not found in evaluation metrics."

        # 初始化最佳指标 / Initialize best metric
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # 训练设置 / Training settings
        self.max_epoch = self.cfg.max_epoch                          # 最大训练轮数 / Maximum epochs
        self.max_iter = self.cfg.max_iter                            # 最大迭代次数 / Maximum iterations
        self.gradient_accumulation_steps = accumulation_steps       # 梯度累积步数 / Gradient accumulation steps
        self.gt_depth_type = self.cfg.gt_depth_type                 # 真实深度数据类型 / GT depth data type
        self.gt_mask_type = self.cfg.gt_mask_type                   # 真实掩码数据类型 / GT mask data type
        self.save_period = self.cfg.trainer.save_period             # 模型保存周期 / Model save period
        self.backup_period = self.cfg.trainer.backup_period         # 备份周期 / Backup period
        self.val_period = self.cfg.trainer.validation_period        # 验证周期 / Validation period
        self.vis_period = self.cfg.trainer.visualization_period     # 可视化周期 / Visualization period

        # 多分辨率噪声配置 / Multi-resolution noise configuration
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength              # 噪声强度 / Noise strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed             # 噪声退火 / Noise annealing
            self.mr_noise_downscale_strategy = (                                   # 下采样策略 / Downscale strategy
                self.cfg.multi_res_noise.downscale_strategy
            )

        # 内部变量初始化 / Internal variables initialization
        self.epoch = 1                              # 当前训练轮数 / Current epoch
        self.n_batch_in_epoch = 0                   # 轮次内批次索引，用于恢复训练 / Batch index in epoch, used when resume training
        self.effective_iter = 0                     # 有效迭代次数（optimizer.step()调用次数） / Effective iterations (times optimizer.step() is called)
        self.in_evaluation = False                  # 是否在评估状态 / Whether in evaluation state
        self.global_seed_sequence: List = []        # 一致的全局种子序列，用于确保恢复时的一致性 / Consistent global seed sequence for consistency when resuming

    def _replace_unet_conv_in(self):
        """
        替换U-Net输入层以接受8个输入通道
        Replace U-Net input layer to accept 8 input channels
        
        将原始的4通道输入层（RGB潜在表示）扩展为8通道（RGB潜在表示 + 噪声深度潜在表示）
        Extend original 4-channel input layer (RGB latent) to 8-channel (RGB latent + noisy depth latent)
        """
        # 获取原始权重和偏置 / Get original weights and bias
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]
        
        # 复制权重以处理8个通道 / Duplicate weights to handle 8 channels
        _weight = _weight.repeat((1, 2, 1, 1))  # 保持选定的通道 / Keep selected channels
        
        # 减半激活幅度以保持数值稳定性 / Half the activation magnitude for numerical stability
        _weight *= 0.5
        
        # 创建新的卷积输入层 / Create new convolution input layer
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        
        # 替换原始层 / Replace original layer
        self.model.unet.conv_in = _new_conv_in
        logging.info("U-Net卷积输入层已替换 / Unet conv_in layer is replaced")
        
        # 更新配置 / Update configuration
        self.model.unet.config["in_channels"] = 8
        logging.info("U-Net配置已更新 / Unet config is updated")
        return

    def train(self, t_end=None):
        """
        主训练循环
        Main training loop
        
        Args:
            t_end: 训练结束时间，用于时间限制训练 / Training end time for time-limited training
        """
        logging.info("开始训练 / Start training")

        device = self.device
        self.model.to(device)

        # 检查是否需要先完成验证 / Check if validation needs to be completed first
        if self.in_evaluation:
            logging.info(
                "上次评估未完成，将在继续训练前进行评估 / Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0  # 梯度累积步数计数器 / Gradient accumulation step counter

        # 主训练循环：轮次遍历 / Main training loop: epoch iteration
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"轮次 / epoch: {self.epoch}")

            # 恢复训练时跳过之前的批次 / Skip previous batches when resuming training
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()  # 设置为训练模式 / Set to training mode

                # 全局一致的随机数生成器 / Globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> 梯度累积开始 / Start of gradient accumulation >>>

                # 获取批次数据 / Get batch data
                rgb = batch["rgb_norm"].to(device)                          # 归一化RGB图像 / Normalized RGB image
                depth_gt_for_latent = batch[self.gt_depth_type].to(device)  # 真实深度数据 / Ground truth depth data

                # 处理有效掩码（如果存在） / Handle valid mask (if exists)
                if self.gt_mask_type is not None:
                    valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                    invalid_mask = ~valid_mask_for_latent
                    # 通过最大池化将掩码下采样到潜在空间分辨率 / Downsample mask to latent space resolution via max pooling
                    valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
                    valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))  # 重复到4个通道 / Repeat to 4 channels

                batch_size = rgb.shape[0]

                # 编码阶段（无梯度） / Encoding phase (no gradient)
                with torch.no_grad():
                    # 编码RGB图像到潜在空间 / Encode RGB image to latent space
                    rgb_latent = self.encode_rgb(rgb)  # [B, 4, h, w]
                    # 编码真实深度到潜在空间 / Encode GT depth to latent space
                    gt_target_latent = self.encode_depth(
                        depth_gt_for_latent
                    )  # [B, 4, h, w]

                # 为每个图像采样随机时间步 / Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # 生成噪声 / Generate noise
                if self.apply_multi_res_noise:
                    # 多分辨率噪声 / Multi-resolution noise
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # 根据时间步计算噪声强度（退火） / Calculate noise strength based on timestep (annealing)
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        gt_target_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    # 标准高斯噪声 / Standard Gaussian noise
                    noise = torch.randn(
                        gt_target_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # 向潜在表示添加噪声（扩散前向过程） / Add noise to latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    gt_target_latent, noise, timesteps
                )  # [B, 4, h, w]

                # 文本嵌入（空文本条件） / Text embedding (empty text conditioning)
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # 连接RGB和目标潜在表示 / Concatenate RGB and target latents
                cat_latents = torch.cat(
                    [rgb_latent, noisy_latents], dim=1
                )  # [B, 8, h, w] - 这是U-Net的输入 / This is U-Net input
                cat_latents = cat_latents.float()

                # 预测噪声残差 / Predict noise residual
                model_pred = self.model.unet(
                    cat_latents, timesteps, text_embed
                ).sample  # [B, 4, h, w]
                
                # 检查NaN值 / Check for NaN values
                if torch.isnan(model_pred).any():
                    logging.warning("模型预测包含NaN / model_pred contains NaN.")

                # 根据预测类型获取损失目标 / Get target for loss based on prediction type
                if "sample" == self.prediction_type:
                    target = gt_target_latent      # 预测原始样本 / Predict original sample
                elif "epsilon" == self.prediction_type:
                    target = noise                 # 预测噪声 / Predict noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(  # 预测速度 / Predict velocity
                        gt_target_latent, noise, timesteps
                    )
                else:
                    raise ValueError(f"未知预测类型 / Unknown prediction type {self.prediction_type}")

                # 计算掩码潜在损失 / Compute masked latent loss
                if self.gt_mask_type is not None:
                    # 仅在有效区域计算损失 / Compute loss only on valid regions
                    latent_loss = self.loss(
                        model_pred[valid_mask_down].float(),
                        target[valid_mask_down].float(),
                    )
                else:
                    # 全局损失 / Global loss
                    latent_loss = self.loss(model_pred.float(), target.float())

                loss = latent_loss.mean()  # 平均损失 / Mean loss

                # 更新训练指标 / Update training metrics
                self.train_metrics.update("loss", loss.item())

                # 梯度累积：损失缩放 / Gradient accumulation: loss scaling
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # 实际批次结束 / Practical batch end

                # 执行优化步骤 / Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()       # 更新参数 / Update parameters
                    self.lr_scheduler.step()    # 更新学习率 / Update learning rate
                    self.optimizer.zero_grad()  # 清零梯度 / Clear gradients
                    accumulated_step = 0

                    self.effective_iter += 1    # 有效迭代计数 / Effective iteration count

                    # 记录到TensorBoard / Log to TensorBoard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dict(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"迭代 {self.effective_iter:5d} (轮次 {epoch:2d}): 损失={accumulated_loss:.5f} / iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # 每步回调函数 / Per-step callback
                    self._train_step_callback()

                    # 训练结束条件检查 / Training end condition checks
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        # 达到最大迭代次数 / Reached maximum iterations
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("训练结束 / Training ended.")
                        return
                    elif t_end is not None and datetime.now() >= t_end:
                        # 时间到达限制 / Time limit reached
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("时间到达，训练暂停 / Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()  # 清理GPU缓存 / Clear GPU cache
                    # <<< 有效批次结束 / Effective batch end <<<

            # 轮次结束 / Epoch end
            self.n_batch_in_epoch = 0

    def encode_rgb(self, image_in):
        """
        编码RGB图像到潜在空间
        Encode RGB image to latent space
        """
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        latent = self.model.encode_rgb(image_in)
        return latent

    def encode_depth(self, depth_in):
        """
        编码深度图到潜在空间
        Encode depth map to latent space
        
        将单通道深度图堆叠为3通道，然后使用VAE编码器进行编码
        Stack single-channel depth to 3-channel, then encode using VAE encoder
        """
        # 将深度图堆叠为3通道 / Stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # 使用VAE编码器编码 / Encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        """
        将深度图堆叠为3通道图像
        Stack depth images to 3-channel images
        
        为了与VAE编码器兼容，需要将单通道深度图重复为3通道
        To be compatible with VAE encoder, single-channel depth needs to be repeated to 3-channel
        """
        if 4 == len(depth_in.shape):
            # [B, 1, H, W] -> [B, 3, H, W]
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            # [1, H, W] -> [1, 3, H, W]
            stacked = depth_in.unsqueeze(1).repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """
        每次迭代后执行的回调函数
        Executed after every iteration
        
        负责：
        1. 保存备份检查点
        2. 执行验证
        3. 保存训练检查点
        4. 执行可视化
        """
        # 保存备份（更大的间隔，不包含训练状态） / Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # 验证 / Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # 设置评估标志，用于恢复运行时的评估检查 / Flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # 保存训练检查点（可以恢复） / Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # 可视化 / Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        """
        验证模型在验证数据集上的性能
        Validate model performance on validation datasets
        
        对每个验证数据集执行推理并计算评估指标
        Perform inference on each validation dataset and compute evaluation metrics
        """
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            
            # 在单个数据集上执行验证 / Validate on single dataset
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            
            logging.info(
                f"迭代 {self.effective_iter}. 在 `{val_dataset_name}` 上的验证指标: {val_metric_dict} / Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
            )
            
            # 记录到TensorBoard / Log to TensorBoard
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )
            
            # 保存到文件 / Save to file
            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dict,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # 更新主要评估指标 / Update main eval metric
            if 0 == i:  # 使用第一个验证数据集的指标作为主要指标 / Use first validation dataset's metric as main metric
                main_eval_metric = val_metric_dict[self.main_val_metric]
                
                # 检查是否达到最佳性能 / Check if best performance is achieved
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"最佳指标: {self.main_val_metric} = {self.best_metric} 在迭代 {self.effective_iter} / Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # 保存最佳模型检查点 / Save best model checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        """
        生成可视化结果
        Generate visualization results
        
        在可视化数据集上执行推理并保存结果图像
        Perform inference on visualization datasets and save result images
        """
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            
            # 在数据集上执行推理并保存可视化结果 / Perform inference and save visualization results
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        """
        在单个数据集上执行验证
        Validate on a single dataset
        
        Args:
            data_loader: 数据加载器 / Data loader
            metric_tracker: 指标追踪器 / Metric tracker
            save_to_dir: 保存目录（可选） / Save directory (optional)
            
        Returns:
            验证指标字典 / Dictionary of validation metrics
        """
        self.model.to(self.device)
        metric_tracker.reset()

        # 生成用于一致评估的种子序列 / Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"在 {data_loader.dataset.disp_name} 上评估 / evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size  # 验证时批次大小必须为1 / Batch size must be 1 for validation
            
            # 读取输入图像 / Read input image
            rgb_int = batch["rgb_int"]  # [B, 3, H, W] - 整数RGB图像 / Integer RGB image
            
            # 获取真实深度和掩码 / Get ground truth depth and mask
            depth_raw_ts = batch["depth_raw_linear"].squeeze()
            depth_raw = depth_raw_ts.numpy()
            depth_raw_ts = depth_raw_ts.to(self.device)
            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            valid_mask = valid_mask_ts.numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)

            # 随机数生成器 / Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # 预测深度 / Predict depth
            pipe_out: MarigoldDepthOutput = self.model(
                rgb_int,
                denoising_steps=self.cfg.validation.denoising_steps,     # 验证时的去噪步数 / Denoising steps for validation
                ensemble_size=self.cfg.validation.ensemble_size,         # 集成大小 / Ensemble size
                processing_res=self.cfg.validation.processing_res,       # 处理分辨率 / Processing resolution
                match_input_res=self.cfg.validation.match_input_res,     # 是否匹配输入分辨率 / Whether to match input resolution
                generator=generator,
                batch_size=1,  # 使用批次大小1以提高可重现性 / Use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,     # 重采样方法 / Resample method
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            # 深度对齐 / Depth alignment
            if "least_square" == self.cfg.eval.alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=depth_raw,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=self.cfg.eval.align_max_res,
                )
            else:
                raise RuntimeError(f"未知对齐类型 / Unknown alignment type: {self.cfg.eval.alignment}")

            # 裁剪到数据集的最小最大值 / Clip to dataset min max
            depth_pred = np.clip(
                depth_pred,
                a_min=data_loader.dataset.min_depth,
                a_max=data_loader.dataset.max_depth,
            )

            # 裁剪到 d > 0 以进行评估 / Clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # 评估 / Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            # 计算所有指标 / Compute all metrics
            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            # 保存为16位uint PNG / Save as 16-bit uint png
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

        return metric_tracker.result()

    def _get_next_seed(self):
        """
        获取下一个种子值
        Get next seed value
        
        从全局种子序列中获取下一个种子，确保训练的可重现性
        Get next seed from global seed sequence to ensure training reproducibility
        """
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"全局种子序列已生成，长度={len(self.global_seed_sequence)} / Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        """
        保存模型检查点
        Save model checkpoint
        
        Args:
            ckpt_name: 检查点名称 / Checkpoint name
            save_train_state: 是否保存训练状态 / Whether to save training state
        """
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"保存检查点到 / Saving checkpoint to: {ckpt_dir}")
        
        # 备份之前的检查点 / Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"旧检查点已备份到 / Old checkpoint is backed up at: {temp_ckpt_dir}")

        # 保存U-Net / Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        logging.info(f"U-Net已保存到 / UNet is saved to: {unet_path}")

        # 保存调度器 / Save scheduler
        scheduelr_path = os.path.join(ckpt_dir, "scheduler")
        self.model.scheduler.save_pretrained(scheduelr_path)
        logging.info(f"调度器已保存到 / Scheduler is saved to: {scheduelr_path}")

        # 保存训练状态（如果需要） / Save training state (if needed)
        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),            # 优化器状态 / Optimizer state
                "lr_scheduler": self.lr_scheduler.state_dict(),      # 学习率调度器状态 / LR scheduler state
                "config": self.cfg,                                 # 配置 / Configuration
                "effective_iter": self.effective_iter,              # 有效迭代次数 / Effective iterations
                "epoch": self.epoch,                                # 当前轮次 / Current epoch
                "n_batch_in_epoch": self.n_batch_in_epoch,          # 轮次内批次数 / Batches in epoch
                "best_metric": self.best_metric,                    # 最佳指标 / Best metric
                "in_evaluation": self.in_evaluation,                # 评估状态 / Evaluation state
                "global_seed_sequence": self.global_seed_sequence,  # 全局种子序列 / Global seed sequence
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            
            # 迭代指示器 / Iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"训练器状态已保存到 / Trainer state is saved to: {train_state_path}")

        # 移除临时检查点 / Remove temp checkpoint
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("旧检查点备份已移除 / Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        """
        加载模型检查点
        Load model checkpoint
        
        Args:
            ckpt_path: 检查点路径 / Checkpoint path
            load_trainer_state: 是否加载训练器状态 / Whether to load trainer state
            resume_lr_scheduler: 是否恢复学习率调度器 / Whether to resume LR scheduler
        """
        logging.info(f"从以下位置加载检查点 / Loading checkpoint from: {ckpt_path}")
        
        # 加载U-Net / Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet.to(self.device)
        logging.info(f"U-Net参数已从以下位置加载 / UNet parameters are loaded from {_model_path}")

        # 加载训练状态 / Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"优化器状态已从以下位置加载 / optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"学习率调度器状态已从以下位置加载 / LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"检查点已从以下位置加载: {ckpt_path}. 从迭代 {self.effective_iter} (轮次 {self.epoch}) 恢复 / Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        """
        获取备份检查点名称
        Get backup checkpoint name
        
        Returns:
            格式化的检查点名称 / Formatted checkpoint name
        """
        return f"iter_{self.effective_iter:06d}"
