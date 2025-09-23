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

import omegaconf
from omegaconf import OmegaConf

# 配置工具函数
# 该模块提供对 OmegaConf/YAML 配置的递归加载与简单的搜索功能：
# - recursive_load_config: 支持通过 base_config 字段递归合并多个配置文件，返回合并后的 OmegaConf 对象
# - find_value_in_omegaconf: 在 DictConfig/ListConfig 中查找指定键名的所有值并以列表返回


def recursive_load_config(config_path: str) -> OmegaConf:
    """
    递归加载并合并配置文件。

    功能：
    - 使用 OmegaConf 加载指定路径的配置文件
    - 如果配置中包含 `base_config` 字段（列表），则递归加载每个 base config，并按顺序合并，
      后加载的配置会覆盖前面的相同字段（与 OmegaConf.merge 的行为一致）

    参数：
    - config_path: 配置文件路径（相对或绝对字符串）

    返回：
    - 合并后的 OmegaConf 对象

    注意：
    - 会检查 base_config 列表中不应包含自身以避免循环合并（会触发断言）
    """
    # 使用 OmegaConf 加载当前配置文件
    conf = OmegaConf.load(config_path)

    # 初始化一个空的配置对象作为合并输出
    output_conf = OmegaConf.create({})

    # 处理 base_config 字段：如果存在，按顺序递归加载并合并每个 base 配置
    base_configs = conf.get("base_config", default_value=None)
    if base_configs is not None:
        # 确保 base_configs 是 ListConfig（OmegaConf 的列表类型）
        assert isinstance(base_configs, omegaconf.listconfig.ListConfig)
        for _path in base_configs:
            # 防止 base_config 中包含自身，导致无限递归
            assert (
                _path != config_path
            ), "Circulate merging, base_config should not include itself."
            # 递归加载父配置并合并到输出中
            _base_conf = recursive_load_config(_path)
            output_conf = OmegaConf.merge(output_conf, _base_conf)

    # 最后将当前配置合并到输出配置中（当前配置优先覆盖已存在字段）
    output_conf = OmegaConf.merge(output_conf, conf)

    return output_conf


def find_value_in_omegaconf(search_key, config):
    """
    在 OmegaConf 的 DictConfig 或 ListConfig 中查找指定键名的所有值。

    行为：
    - 遍历 DictConfig 的键值对，若键等于 search_key，收集对应的 value
    - 对于值仍为 DictConfig 或 ListConfig 的子结构，递归搜索
    - 对 ListConfig 中的每个元素，若该元素是 DictConfig 或 ListConfig，也递归搜索

    返回：
    - 包含所有匹配值的列表（可能为空）

    说明：这个函数仅按键名匹配键，不会比较完整路径；如果配置中存在多个同名键，
    它会返回所有对应的值（按遍历顺序）。
    """
    result_list = []

    # 如果是 DictConfig，遍历键值对
    if isinstance(config, omegaconf.DictConfig):
        for key, value in config.items():
            if key == search_key:
                result_list.append(value)
            # 如果值是嵌套的字典或列表配置，继续递归查找
            elif isinstance(value, (omegaconf.DictConfig, omegaconf.ListConfig)):
                result_list.extend(find_value_in_omegaconf(search_key, value))
    # 如果是 ListConfig，对每个元素递归搜索（当元素是 DictConfig 或 ListConfig 时）
    elif isinstance(config, omegaconf.ListConfig):
        for item in config:
            if isinstance(item, (omegaconf.DictConfig, omegaconf.ListConfig)):
                result_list.extend(find_value_in_omegaconf(search_key, item))

    return result_list


if "__main__" == __name__:
    conf = recursive_load_config("config/train_base.yaml")
    print(OmegaConf.to_yaml(conf))
