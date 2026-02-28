"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

from point_injetctor import PointVLAInjector

# 导入您项目中 PointNet++ 的定义
from pointnetpp.pointnetpp import PointNetPP
from torch.nn import CrossEntropyLoss

# 导入Hugging Face的标准输出格式，这是之前遗漏的
from transformers.modeling_outputs import CausalLMOutputWithPast

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    config_class = OpenVLAConfig  # <-- ADD THIS LINE
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids,
            config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # === START: PointVLA Architectural Changes ===

        # 1. 实例化一个可训练的 PointNet++ 编码器 (从零开始训练)
        #    这里的架构可以自定义，输出一个全局特征向量
        point_cloud_feature_dim = 768
        self.pointnet_encoder = PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, point_cloud_feature_dim]],
            target_dim=point_cloud_feature_dim,  # <-- 添加这个缺失的参数
            return_feature_sequence=False  # <-- 明确指定我们不需要特征序列
        )

        # 2. 实例化 PointVLA 注入器
        llm_hidden_dim = self.config.text_config.hidden_size
        self.pointvla_injector = PointVLAInjector(
            num_injection_blocks=5,
            llm_hidden_dim=llm_hidden_dim,
            point_cloud_feature_dim=point_cloud_feature_dim
        )

        # 3. 定义注入点 (根据您的LLM层数调整)
        #    例如，对于 Llama-7B (32层)，选择11层之后的5个点
        self.injection_points = [12, 15, 18, 21, 24]

        # === END: PointVLA Architectural Changes ===

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
            self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            pointcloud: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # --- 初始化标准Hugging Face输出控制 ---
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # === Step 1: 准备多模态嵌入 (文本 + 图像) ===
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        num_vision_tokens = 0
        # 仅在第一次前向传播时（即没有缓存时）处理视觉输入
        if past_key_values is None:
            patch_features = self.vision_backbone(pixel_values)
            projected_patch_embeddings = self.projector(patch_features)
            num_vision_tokens = projected_patch_embeddings.shape[1]
            multimodal_embeddings = torch.cat(
                [inputs_embeds[:, :1, :], projected_patch_embeddings, inputs_embeds[:, 1:, :]],
                dim=1
            )

            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    True, dtype=attention_mask.dtype, device=attention_mask.device
                )
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]],
                    dim=1
                )
            else:
                multimodal_attention_mask = None
        else:
            # 如果有缓存，说明是生成后续token，直接使用 input_ids (通常只有一个token)
            multimodal_embeddings = inputs_embeds
            multimodal_attention_mask = attention_mask

        # === Step 2: 提取并扩展点云特征 (仅在首次传播时) ===
        pcd_features_for_injection = None
        if pointcloud is not None and hasattr(self, 'pointnet_encoder'):
            with torch.autocast(device_type=pointcloud.device.type, enabled=False):
                pointcloud_fp32 = pointcloud.to(torch.float32)
                pointnet_fp32 = self.pointnet_encoder.to(torch.float32)
                pointnet_fp32.train()  # 解决BN问题
                pcd_features_global_fp32 = pointnet_fp32(pointcloud_fp32)

            # 扩展为序列
            N_EXPAND = 16
            pcd_features_expanded_fp32 = pcd_features_global_fp32.repeat(1, N_EXPAND, 1)
            pcd_features_for_injection = pcd_features_expanded_fp32.to(self.dtype)

        # === Step 3: 手动执行Llama前向传播并注入特征 ===
        hidden_states = multimodal_embeddings

        device = hidden_states.device
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        seq_length = hidden_states.shape[1]
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)
        expanded_multimodal_attention_mask = None
        if multimodal_attention_mask is not None:
            expanded_multimodal_attention_mask = \
                self.language_model.model._update_causal_mask(
                    attention_mask=multimodal_attention_mask,
                    input_tensor=multimodal_embeddings,
                    cache_position=position_ids,
                    past_seen_tokens=past_key_values_length
                )


        # --- 恢复完整的缓存和输出控制逻辑 ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        llm_blocks = self.language_model.model.layers

        injector_block_idx = 0
        for i, layer_module in enumerate(llm_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 为每一层准备缓存
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=expanded_multimodal_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # 在指定注入点注入点云特征
            if (i + 1) in self.injection_points and pcd_features_for_injection is not None:
                hidden_states = self.pointvla_injector(
                    hidden_states,
                    pcd_features_for_injection,
                    injector_block_idx
                )
                injector_block_idx += 1

        # === Step 4: 计算最终输出和损失 ===
        hidden_states = self.language_model.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.language_model.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # --- START: FIX for labels alignment ---
            # The `labels` tensor has the length of text tokens only.
            # The `logits` tensor has the length of the full multimodal sequence.
            # We must align them by padding the `labels` with IGNORE_INDEX for the vision tokens.
            vision_labels = torch.full(
                (labels.shape[0], num_vision_tokens), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )

            # The multimodal embeddings were assembled as: [BOS, VISION, TEXT]
            # We align the labels in the same way: [BOS_LABEL, IGNORE_LABELS, TEXT_LABELS]
            aligned_labels = torch.cat([labels[:, :1], vision_labels, labels[:, 1:]], dim=1)
            # --- END: FIX ---

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = aligned_labels[..., 1:].contiguous()  # <-- USE THE NEW ALIGNED LABELS

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + (next_decoder_cache,) + all_hidden_states + all_self_attns
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # ===============================================================================
    # ===================== END: 完整且正确的 FORWARD 方法 =====================
    # ===============================================================================

    # === Core Prismatic VLM `forward()` Logic ===
    # def forward(
    #         self,
    #         input_ids: Optional[torch.LongTensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         pixel_values: Optional[torch.FloatTensor] = None,
    #         pointcloud: Optional[torch.Tensor] = None,
    #         labels: Optional[torch.LongTensor] = None,
    #         inputs_embeds: Optional[torch.FloatTensor] = None,
    #         past_key_values: Optional[List[torch.FloatTensor]] = None,
    #         use_cache: Optional[bool] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         output_projector_features: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
    #     """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     output_projector_features = output_projector_features if output_projector_features is not None else False
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
    #     use_cache = use_cache and not self.training
    #
    #     # Instantiate Placeholder for Projector Features
    #     projected_patch_embeddings = None
    #
    #     # Branching Logic:
    #     #   - If `past_key_values` are provided, we're doing cached generation. This is the Causal LM forward pass.
    #     #   - Otherwise, we're doing a full forward pass.
    #     #       - If `pixel_values` are not provided, this is a language-only forward pass.
    #     #       - If `pixel_values` are provided, this is a multimodal forward pass.
    #     if past_key_values is not None:
    #         # Important: during cached generation, `pixel_values` and `pointcloud` should be None
    #         assert pixel_values is None, "pixel_values should be None when past_key_values are provided"
    #         assert pointcloud is None, "pointcloud should be None when past_key_values are provided"
    #
    #         # Standard Causal LM forward pass for generation
    #         language_model_output = self.language_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=None,
    #             past_key_values=past_key_values,
    #             inputs_embeds=None,
    #             labels=None,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #
    #     elif pixel_values is None:
    #         # Unimodal (language-only) forward pass
    #         language_model_output = self.language_model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=inputs_embeds,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #
    #     else:
    #         # Multimodal forward pass (initial processing)
    #         assert (input_ids is not None) or (inputs_embeds is not None)
    #
    #         # Visual Feature Extraction
    #         patch_features = self.vision_backbone(pixel_values)
    #         projected_patch_embeddings = self.projector(patch_features)
    #
    #         # Get Input Embeddings
    #         if inputs_embeds is None:
    #             inputs_embeds = self.get_input_embeddings()(input_ids)
    #
    #         # --- Conditional Point Cloud Processing ---
    #         if pointcloud is not None:
    #             # 1. 确保输入点云是 float32 类型
    #             pointcloud_fp32 = pointcloud.to(torch.float32)
    #
    #             # 2. 临时将 pointnet 模块自身转换为 float32
    #             pointnet_fp32 = self.pointnet_encoder.to(torch.float32)
    #             pointnet_fp32.train()
    #             a = list(pointnet_fp32.parameters())
    #             # 3. 在 float32 精度下执行计算
    #             pcd_features_fp32 = pointnet_fp32(pointcloud_fp32)
    #
    #             # 4. 将输出特征转换回模型主体的 bfloat16 类型，以进行后续拼接
    #             pcd_features = pcd_features_fp32.to(self.dtype)
    #
    #             # Combine Embeddings: BOS | IMAGE | POINTCLOUD | TEXT
    #             multimodal_embeddings = torch.cat(
    #                 [inputs_embeds[:, :1, :], projected_patch_embeddings, pcd_features, inputs_embeds[:, 1:, :]],
    #                 dim=1
    #             )
    #
    #             # Create extended attention mask and labels
    #             if attention_mask is not None:
    #                 projected_patch_attention_mask = torch.full(
    #                     (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #                     True, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 pcd_attention_mask = torch.full(
    #                     (pcd_features.shape[0], pcd_features.shape[1]),
    #                     True, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 attention_mask = torch.cat(
    #                     [attention_mask[:, :1], projected_patch_attention_mask, pcd_attention_mask,
    #                      attention_mask[:, 1:]], dim=1)
    #
    #             if labels is not None:
    #                 projected_patch_labels = torch.full(
    #                     (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #                     IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
    #                 pcd_labels = torch.full(
    #                     (pcd_features.shape[0], pcd_features.shape[1]),
    #                     IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
    #                 labels = torch.cat([labels[:, :1], projected_patch_labels, pcd_labels, labels[:, 1:]], dim=1)
    #
    #         else:  # Image-only multimodal forward pass
    #             multimodal_embeddings = torch.cat(
    #                 [inputs_embeds[:, :1, :], projected_patch_embeddings, inputs_embeds[:, 1:, :]], dim=1)
    #
    #             if attention_mask is not None:
    #                 projected_patch_attention_mask = torch.full(
    #                     (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #                     True, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 attention_mask = torch.cat(
    #                     [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1)
    #
    #             if labels is not None:
    #                 projected_patch_labels = torch.full(
    #                     (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    #                     IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
    #                 labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
    #
    #         # Dispatch to Language Model
    #         language_model_output = self.language_model(
    #             input_ids=None,
    #             attention_mask=attention_mask,
    #             position_ids=None,
    #             past_key_values=None,
    #             inputs_embeds=multimodal_embeddings,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #
    #     if not return_dict:
    #         if output_projector_features and (projected_patch_embeddings is not None):
    #             return *language_model_output, projected_patch_embeddings
    #         return language_model_output
    #
    #     return PrismaticCausalLMOutputWithPast(
    #         loss=language_model_output.loss,
    #         logits=language_model_output.logits,
    #         past_key_values=language_model_output.past_key_values,
    #         hidden_states=language_model_output.hidden_states,
    #         attentions=language_model_output.attentions,
    #         projector_features=projected_patch_embeddings,
    #     )
    #
    # # === GenerationMixin Methods ===
    # # def prepare_inputs_for_generation(
    # #     self,
    # #     input_ids: Optional[torch.Tensor] = None,
    # #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    # #     inputs_embeds: Optional[torch.FloatTensor] = None,
    # #     pixel_values: Optional[torch.FloatTensor] = None,
    # #     attention_mask: Optional[torch.Tensor] = None,
    # #     **kwargs: str,
    # # ) -> Dict[str, torch.Tensor]:
    # #     """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
    # #     if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
    # #         (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
    # #     ):
    # #         raise ValueError("Generation with batch size > 1 is not currently supported!")
    # #
    # #     # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
    # #     if past_key_values is not None:
    # #         input_ids = input_ids[:, -1:]
    # #
    # #     # If `input_embeds` are passed, we only want to use them in the 1st generation step
    # #     if inputs_embeds is not None and past_key_values is None:
    # #         model_inputs = {"input_embeds": inputs_embeds}
    # #     else:
    # #         model_inputs = {"input_ids": input_ids}
    # #
    # #     # Make sure `pixel_values` are preserved in `model_inputs`
    # #     model_inputs.update(
    # #         {
    # #             "attention_mask": attention_mask,
    # #             "pixel_values": pixel_values,
    # #             "past_key_values": past_key_values,
    # #             "use_cache": kwargs.get("use_cache"),
    # #         }
    # #     )
    # #
    # #     return model_inputs
    def prepare_inputs_for_generation(self, *args, **kwargs):
        # 这个方法确保在 .generate() 调用中，缓存被正确处理
        # 它将视觉输入（pixel_values, pointcloud）与 past_key_values 解耦

        # 获取点云（如果存在于kwargs中）
        pointcloud = kwargs.get("pointcloud", None)

        # 调用原始语言模型的 prepare_inputs_for_generation 来处理 input_ids, past_key_values 等
        model_inputs = self.language_model.prepare_inputs_for_generation(*args, **kwargs)

        # 如果没有缓存（第一次调用），则保留 pixel_values 和 pointcloud
        # 如果有缓存，则它们应为 None，由原始方法处理
        if model_inputs.get('past_key_values') is None:
            model_inputs['pixel_values'] = kwargs.get('pixel_values')
            model_inputs['pointcloud'] = pointcloud

        return model_inputs
    # def prepare_inputs_for_generation(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         past_key_values: Optional[List[torch.FloatTensor]] = None,
    #         inputs_embeds: Optional[torch.FloatTensor] = None,
    #         pixel_values: Optional[torch.FloatTensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         **kwargs: str,
    # ) -> Dict[str, torch.Tensor]:
    #     """Prepares inputs for generation. If a cache is passed, nullifies visual inputs to prevent re-computation."""
    #     if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
    #             (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
    #     ):
    #         raise ValueError("Generation with batch size > 1 is not currently supported!")
    #
    #     pointcloud = kwargs.get("pointcloud", None)
    #
    #     # ==================== START: KEY FIX ====================
    #     # If we have a cache (`past_key_values`), we're in the middle of autoregressive decoding.
    #     # We only need the last token `input_ids` and the cache.
    #     # The visual inputs (`pixel_values`, `pointcloud`) should be set to None to prevent re-encoding.
    #     if past_key_values is not None:
    #         input_ids = input_ids[:, -1:]
    #         pixel_values = None
    #         pointcloud = None
    #     # ===================== END: KEY FIX =====================
    #
    #     # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and past_key_values is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids}
    #
    #     # Update the model inputs with all necessary items for the forward pass
    #     model_inputs.update(
    #         {
    #             "attention_mask": attention_mask,
    #             "pixel_values": pixel_values,
    #             "pointcloud": pointcloud,
    #             "past_key_values": past_key_values,
    #             "use_cache": kwargs.get("use_cache"),
    #         }
    #     )
    #
    #     return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    # def predict_action(
    #     self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    # ) -> np.ndarray:
    #     """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
    #     # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    #     # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    #     if not torch.all(input_ids[:, -1] == 29871):
    #         input_ids = torch.cat(
    #             (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
    #         )
    #
    #     # Run VLA inference
    #     generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
    #
    #     # Extract predicted action tokens and translate into (normalized) continuous actions
    #     predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
    #     discretized_actions = self.vocab_size - predicted_action_token_ids
    #     discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
    #     normalized_actions = self.bin_centers[discretized_actions]
    #
    #     # Unnormalize actions
    #     action_norm_stats = self.get_action_stats(unnorm_key)
    #     mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    #     action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    #     actions = np.where(
    #         mask,
    #         0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
    #         normalized_actions,
    #     )
    #
    #     return actions
    def predict_action(
            self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # 如果 kwargs 中包含 pointcloud，将其提取出来
        pointcloud = kwargs.pop("pointcloud", None)

        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        generated_ids = self.generate(
            input_ids,
            pointcloud=pointcloud,  # <== 将点云数据传递给 generate
            max_new_tokens=self.get_action_dim(unnorm_key),
            **kwargs
        )

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key):].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
