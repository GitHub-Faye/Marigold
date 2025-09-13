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

import logging
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Union

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)


class MarigoldDepthOutput(BaseOutput):
    """
    Marigold单目深度估计管道的输出类。
    Output class for Marigold Monocular Depth Estimation pipeline.

    Args:
        depth_np (`np.ndarray`):
            预测的深度图，深度值范围为[0, 1]。
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            着色深度图，形状为[H, W, 3]，值在[0, 255]范围内。
            Colorized depth map, with the shape of [H, W, 3] and values in [0, 255].
        uncertainty (`None` or `np.ndarray`):
            来自集成的未校准不确定性（MAD，中位绝对偏差）。
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MarigoldDepthPipeline(DiffusionPipeline):
    """
    Marigold单目深度估计管道：https://marigoldcomputervision.github.io。
    Pipeline for Marigold Monocular Depth Estimation: https://marigoldcomputervision.github.io.

    此模型继承自[`DiffusionPipeline`]。查看超类文档了解库为所有管道实现的通用方法（如下载、保存、在特定设备上运行等）。
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            条件U-Net，用于对预测潜在表示进行去噪，以图像潜在表示为条件。
            Conditional U-Net to denoise the prediction latent, conditioned on image latent.
        vae (`Union[AutoencoderKL, AutoencoderTiny]`):
            变分自编码器（VAE）模型，用于将图像和预测编码和解码为潜在表示。可以是AutoencoderKL或AutoencoderTiny。
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions
            to and from latent representations. Can be either AutoencoderKL or AutoencoderTiny.
        scheduler (`DDIMScheduler`):
            与`unet`结合使用的调度器，用于对编码的图像潜在表示进行去噪。
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            文本编码器，用于空文本嵌入。
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP分词器。
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            模型属性，指定预测的深度图是否具有尺度不变性。此值必须在模型配置中设置。
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            模型属性，指定预测的深度图是否具有平移不变性。此值必须在模型配置中设置。
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            生成合理质量预测所需的最少去噪扩散步数。此值必须在模型配置中设置。
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            管道`processing_resolution`参数的推荐值。此值必须在模型配置中设置。
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    latent_scale_factor = 0.18215  # AutoencoderKL的默认缩放因子 / default scaling factor for AutoencoderKL 

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        # 注册所有模型组件 / Register all model components
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        # 注册配置参数 / Register configuration parameters
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        # 保存配置属性 / Store configuration attributes
        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        # 空文本嵌入缓存 / Empty text embedding cache
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, *optional*, defaults to `1`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize the prediction to match the input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or
                `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False,
                near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [H, W, 3] and values in [0, 255], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # 模型特定的最优默认值，可获得快速且合理的结果 / Model-specific optimal default values
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # 检查去噪步数是否合理 / Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- 图像预处理 / Image Preprocess -----------------
        # 转换为torch张量 / Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # 转换为torch张量 [H, W, rgb] -> [rgb, H, W] / convert to torch tensor
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # 调整图像大小 / Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # 标准化RGB值 / Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- 预测深度 / Predicting depth -----------------
        # 批量重复输入图像 / Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # 批量预测深度图 / Predict depth maps (batched)
        target_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            target_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            target_pred_ls.append(target_pred_raw.detach())
        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()  # 清空显存缓存以进行集成 / clear vram cache for ensembling

        # ----------------- 测试时集成 / Test-time ensembling -----------------
        if ensemble_size > 1:
            # 集成多个深度预测 / Ensemble multiple depth predictions
            final_pred, pred_uncert = ensemble_depth(
                target_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                **(ensemble_kwargs or {}),
            )
        else:
            final_pred = target_preds
            pred_uncert = None

        # 调整回原始分辨率 / Resize back to original resolution
        if match_input_res:
            final_pred = resize(
                final_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # 转换为numpy数组 / Convert to numpy
        final_pred = final_pred.squeeze()
        final_pred = final_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # 裁剪输出范围 / Clip output range
        final_pred = final_pred.clip(0, 1)

        # 着色处理 / Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                final_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=final_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        检查去噪步数是否合理
        Check if denoising step is reasonable
        Args:
            n_step (`int`): 去噪步数 / denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            logging.warning(
                "DeprecationWarning: LCMScheduler will not be supported in the future. "
                "Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
            )
            if n_step > 10:
                logging.warning(
                    f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                    f"the default values."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        为空提示编码文本嵌入
        Encode text embedding for empty prompt
        """
        prompt = ""  # 空提示 / Empty prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        # 生成并缓存空文本嵌入 / Generate and cache empty text embedding
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        执行单次预测，不进行集成。
        Perform a single prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                输入RGB图像。
                Input RGB image.
            num_inference_steps (`int`):
                推理期间的扩散去噪步数（DDIM）。
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (`bool`):
                显示扩散去噪的进度条。
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                用于初始噪声生成的随机生成器。
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: 预测目标。
            `torch.Tensor`: Predicted targets.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # 设置时间步 / Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # 编码图像 / Encode image
        rgb_latent = self.encode_rgb(rgb_in)  # [B, 4, h, w]

        # 用于输出的噪声潜在表示 / Noisy latent for outputs
        target_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, target_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            target_latent = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            ).prev_sample

        depth = self.decode_depth(target_latent)  # [B,3,H,W]

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        将RGB图像编码为潜在表示。
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                要编码的输入RGB图像。
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: 图像潜在表示。
            `torch.Tensor`: Image latent.
        """
        # 通过类名检查VAE类型以获得更可靠的检测 / Check VAE type by class name for more reliable detection
        if self.vae.__class__.__name__ == 'AutoencoderTiny':
            # AutoencoderTiny路径 / AutoencoderTiny path
            rgb_latent = self.vae.encode(rgb_in).latents * self.vae.config.scaling_factor
            # TAESD scaling_factor=1.0, AutoencoderKL scaling_factor=0.18215
        else:
            # AutoencoderKL路径 / AutoencoderKL path
            h = self.vae.encoder(rgb_in)
            moments = self.vae.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            # 缩放潜在表示 / scale latent
            rgb_latent = mean * self.latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        将深度潜在表示解码为深度图。
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                要解码的深度潜在表示。
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: 解码的深度图。
            `torch.Tensor`: Decoded depth map.
        """

        # 通过类名检查VAE类型以获得更可靠的检测 / Check VAE type by class name for more reliable detection
        if self.vae.__class__.__name__ == 'AutoencoderTiny':
            # AutoencoderTiny路径 / AutoencoderTiny path
            # AutoencoderKL scaling_factor=0.18215, TAESD scaling_factor=1.0
            depth_latent_taesd = depth_latent / self.vae.config.scaling_factor 
            decoded = self.vae.decode(depth_latent_taesd).sample
            depth_mean = decoded.mean(dim=1, keepdim=True)
        else:
            # scale latent
            depth_latent = depth_latent / self.latent_scale_factor
            # AutoencoderKL路径 / AutoencoderKL path
            z = self.vae.post_quant_conv(depth_latent)
            stacked = self.vae.decoder(z)
            # 输出通道的均值 / mean of output channels
            depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
