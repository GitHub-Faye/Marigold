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
Marigold深度估计模型训练脚本
Marigold Depth Estimation Training Script

主要功能：
1. 配置文件加载和命令行参数解析
2. 训练环境初始化（设备、日志、数据路径）
3. 数据集加载和数据加载器创建
4. 模型初始化和检查点管理
5. 训练器初始化和训练循环执行
"""

import sys
import os

# 添加项目根目录到Python路径 / Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import os
import shutil
import torch
from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from typing import List, Union

# Marigold核心组件导入 / Import Marigold core components
from marigold import MarigoldDepthPipeline
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_trainer_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
)
from src.util.slurm_util import get_local_scratch_dir, is_on_slurm


if "__main__" == __name__:
    # 记录训练开始时间 / Record training start time
    t_start = datetime.now()
    logging.info(f"Started at {t_start}")

    # -------------------- 命令行参数解析 / Arguments Parsing --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : 单目深度估计 : 训练 / Monocular Depth Estimation : Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_marigold_depth.yaml",
        help="配置文件路径 / Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="要恢复的检查点路径。如果提供，将忽略--config和配置中的检查点 / Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="保存检查点的目录 / Directory to save checkpoints."
    )
    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        help="不使用cuda / Do not use cuda."
    )
    parser.add_argument(
        "--exit_after",
        type=int,
        default=-1,
        help="X分钟后保存检查点并退出 / Save checkpoint and exit after X minutes.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="不使用Weights and Biases日志记录 / Run without Weights and Biases logging.",
    )
    parser.add_argument(
        "--do_not_copy_data",
        action="store_true",
        help="在Slurm集群上，不将数据复制到本地临时目录 / On Slurm cluster, do not copy data to the local scratch.",
    )
    parser.add_argument(
        "--base_data_dir", 
        type=str, 
        default=None, 
        help="数据集基础路径 / Base path to the datasets."
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="预训练检查点基础路径 / Base path to the pretrained checkpoints.",
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_true",
        help="在输出文件夹名称中添加日期时间 / Add datetime to the output folder name.",
    )

    # 解析命令行参数 / Parse command line arguments
    args = parser.parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    
    # 设置数据和检查点基础目录 / Set base directories for data and checkpoints
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )

    # -------------------- 训练环境初始化 / Training Environment Initialization --------------------
    
    # 恢复之前的训练运行 / Resume previous training run
    if resume_run is not None:
        logging.info(f"恢复训练运行: {resume_run} / Resuming run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # 恢复配置文件 / Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # 从头开始训练 / Run from start
        cfg = recursive_load_config(args.config)
        # 完整任务名称 / Full job name
        pure_job_name = os.path.basename(args.config).split(".")[0]
        
        # 添加时间前缀 / Add time prefix
        if args.add_datetime_prefix:
            job_name = f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}-{pure_job_name}"
        else:
            job_name = pure_job_name

        # 输出目录 / Output directory
        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        os.makedirs(out_dir_run, exist_ok=False)

    cfg_data = cfg.dataset

    # 创建其他输出目录 / Create other output directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")  # 检查点目录 / Checkpoint directory
    if not os.path.exists(out_dir_ckpt):
        os.makedirs(out_dir_ckpt)
        
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")  # TensorBoard日志目录 / TensorBoard log directory
    if not os.path.exists(out_dir_tb):
        os.makedirs(out_dir_tb)
        
    out_dir_eval = os.path.join(out_dir_run, "evaluation")  # 评估结果目录 / Evaluation results directory
    if not os.path.exists(out_dir_eval):
        os.makedirs(out_dir_eval)
        
    out_dir_vis = os.path.join(out_dir_run, "visualization")  # 可视化结果目录 / Visualization results directory
    if not os.path.exists(out_dir_vis):
        os.makedirs(out_dir_vis)

    # -------------------- 日志设置 / Logging Settings --------------------
    config_logging(cfg.logging, out_dir=out_dir_run)
    logging.debug(f"配置信息 / config: {cfg}")

    # 初始化wandb实验追踪 / Initialize wandb experiment tracking
    if not args.no_wandb:
        if resume_run is not None:
            # 恢复已有wandb运行 / Resume existing wandb run
            wandb_id = load_wandb_job_id(out_dir_run)
            wandb_cfg_dict = {
                "id": wandb_id,
                "resume": "must",
                **cfg.wandb,
            }
        else:
            # 创建新的wandb运行 / Create new wandb run
            wandb_cfg_dict = {
                "config": dict(cfg),
                "name": job_name,
                "mode": "online",
                **cfg.wandb,
            }
        wandb_cfg_dict.update({"dir": out_dir_run})
        wandb_run = init_wandb(enable=True, **wandb_cfg_dict)
        save_wandb_job_id(wandb_run, out_dir_run)
    else:
        init_wandb(enable=False)

    # TensorBoard日志记录（应该在wandb之后初始化） / Tensorboard logging (should be initialized after wandb)
    tb_logger.set_dir(out_dir_tb)

    # 记录Slurm作业ID / Log Slurm job ID
    log_slurm_job_id(step=0)

    # -------------------- 设备配置 / Device Configuration --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"使用设备 / device = {device}")

    # -------------------- 代码和配置快照 / Snapshot of code and config --------------------
    if resume_run is None:
        # 保存配置文件 / Save config file
        _output_path = os.path.join(out_dir_run, "config.yaml")
        with open(_output_path, "w+") as f:
            OmegaConf.save(config=cfg, f=f)
        logging.info(f"配置已保存到 / Config saved to {_output_path}")
        
        # 在第一次运行时复制和打包代码 / Copy and tar code on the first run
        _temp_code_dir = os.path.join(out_dir_run, "code_tar")
        _code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logging.info(f"代码快照已保存到 / Code snapshot saved to: {_code_snapshot_path}")

    # -------------------- 将数据复制到本地临时目录（Slurm集群） / Copy data to local scratch (Slurm) --------------------
    if is_on_slurm() and (not args.do_not_copy_data):
        # 本地临时目录 / local scratch dir
        original_data_dir = base_data_dir
        base_data_dir = os.path.join(get_local_scratch_dir(), "Marigold_data")
        
        # 复制数据 / copy data
        required_data_list = find_value_in_omegaconf("dir", cfg_data)
        required_data_list = list(set(required_data_list))
        logging.info(f"所需数据列表 / Required_data_list: {required_data_list}")
        
        for d in tqdm(required_data_list, desc="将数据复制到本地临时目录 / Copy data to local scratch"):
            ori_dir = os.path.join(original_data_dir, d)
            dst_dir = os.path.join(base_data_dir, d)
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            if os.path.isfile(ori_dir):
                shutil.copyfile(ori_dir, dst_dir)
            elif os.path.isdir(ori_dir):
                shutil.copytree(ori_dir, dst_dir)
        logging.info(f"数据已复制到 / Data copied to: {base_data_dir}")

    # -------------------- 梯度累积步数配置 / Gradient accumulation steps configuration --------------------
    eff_bs = cfg.dataloader.effective_batch_size  # 有效批次大小 / Effective batch size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size  # 累积步数 / Accumulation steps
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    logging.info(
        f"有效批次大小: {eff_bs}, 梯度累积步数: {accumulation_steps} / Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}"
    )

    # -------------------- 数据加载配置 / Data Loading Configuration --------------------
    
    # 数据加载器随机种子设置 / Dataloader random seed setting
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # 训练数据集 / Training dataset
    depth_transform: DepthNormalizerBase = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )
    train_dataset: Union[BaseDepthDataset, List[BaseDepthDataset]] = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )
    logging.debug("数据增强配置 / Augmentation: ", cfg.augmentation)
    
    # 处理混合数据集或单一数据集 / Handle mixed dataset or single dataset
    if "mixed" == cfg_data.train.name:
        # 混合数据集：多个数据集按概率采样 / Mixed dataset: multiple datasets with probability sampling
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "长度不匹配: `prob_ls` 和 `dataset_list` / Lengths don't match: `prob_ls` and `dataset_list`"
        
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        # 单一数据集 / Single dataset
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            generator=loader_generator,
        )
        
    # 验证数据集 / Validation dataset
    val_loaders: List[DataLoader] = []
    for _val_dict in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dict,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,  # 验证时批次大小为1 / Batch size 1 for validation
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        val_loaders.append(_val_loader)

    # 可视化数据集 / Visualization dataset
    vis_loaders: List[DataLoader] = []
    for _vis_dict in cfg_data.vis:
        _vis_dataset = get_dataset(
            _vis_dict,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,  # 可视化时批次大小为1 / Batch size 1 for visualization
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        vis_loaders.append(_vis_loader)

    # -------------------- 模型初始化 / Model Initialization --------------------
    
    # 管道参数配置 / Pipeline kwargs configuration
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    
    # 加载预训练的Marigold深度估计管道 / Load pretrained Marigold depth estimation pipeline
    model = MarigoldDepthPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path), 
        **_pipeline_kwargs
    )

    # -------------------- 训练器初始化 / Trainer Initialization --------------------
    
    # 退出时间设置 / Exit time setting
    if args.exit_after > 0:
        t_end = t_start + timedelta(minutes=args.exit_after)
        logging.info(f"将在 {t_end} 退出 / Will exit at {t_end}")
    else:
        t_end = None

    # 获取训练器类 / Get trainer class
    trainer_cls = get_trainer_cls(cfg.trainer.name)
    logging.debug(f"训练器类 / Trainer: {trainer_cls}")
    
    # 初始化训练器 / Initialize trainer
    trainer = trainer_cls(
        cfg=cfg,                          # 配置文件 / Configuration
        model=model,                      # 模型 / Model
        train_dataloader=train_loader,    # 训练数据加载器 / Training dataloader
        device=device,                    # 计算设备 / Compute device
        out_dir_ckpt=out_dir_ckpt,       # 检查点输出目录 / Checkpoint output directory
        out_dir_eval=out_dir_eval,       # 评估结果输出目录 / Evaluation output directory
        out_dir_vis=out_dir_vis,         # 可视化结果输出目录 / Visualization output directory
        accumulation_steps=accumulation_steps,  # 梯度累积步数 / Gradient accumulation steps
        val_dataloaders=val_loaders,     # 验证数据加载器列表 / Validation dataloaders
        vis_dataloaders=vis_loaders,     # 可视化数据加载器列表 / Visualization dataloaders
    )

    # -------------------- 检查点管理 / Checkpoint Management --------------------
    if resume_run is not None:
        # 加载检查点以恢复训练 / Load checkpoint to resume training
        trainer.load_checkpoint(
            resume_run, 
            load_trainer_state=True,      # 加载训练器状态 / Load trainer state
            resume_lr_scheduler=True      # 恢复学习率调度器 / Resume learning rate scheduler
        )

    # -------------------- 训练与评估循环 / Training & Evaluation Loop --------------------
    try:
        # 开始训练 / Start training
        trainer.train(t_end=t_end)
    except Exception as e:
        # 记录异常信息 / Log exception information
        logging.exception(e)
