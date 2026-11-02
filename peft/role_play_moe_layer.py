# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import warnings
import os
import json
import pickle
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import (
    dequantize_module_weight,
    gather_params_ctx,
    get_bnb_param_type,
    skip_init_on_device,
)
from peft.utils.other import transpose
from peft.utils.warning import PeftWarning

from .config import ArrowConfig, LoraConfig

from loguru import logger
from tqdm import tqdm


VARIANT_KWARG_KEYS = ["alora_offsets"]


class LoraVariant:
    """
    Base class for LoRA variants, e.g. DoRA.

    This class should be subclassed and the methods below should be implemented accordingly. The methods should be
    implemented as static methods, this makes it easier to combine variants.

    Note for developers: These methods are prone to change and should thus considered to be "private". Use at your own
    discretion.
    """

    @staticmethod
    def init(module: LoraLayer, adapter_name: str) -> None:
        """Initialization code for the LoRA variant, it's called within `update_layer`"""
        raise NotImplementedError

    @staticmethod
    def merge_safe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """Safe merging of the weights from `merge(..., safe_merge=True)`, should return a new tensor"""
        raise NotImplementedError

    @staticmethod
    def merge_unsafe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> None:
        """Unsafe merging of the weights from `merge(..., safe_merge=False)`, should modify the weight in-place"""

    @staticmethod
    def unmerge(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """Remove the adapter weights from the original weights, then return them"""

    @staticmethod
    def forward(
        module: LoraLayer,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        The forward pass of the LoRA variant, should return the overall result (not just the diff)

        Args:
            module (LoraLayer): The module on which the forward pass is called
            active_adapter (str): The name of the active adapter
            x (torch.Tensor): The input to the forward call
            result (torch.Tensor): The result from the base model
            **kwargs: Additional arguments passed from [`LoraLayer.forward`].
        """
        raise NotImplementedError


class Expert(nn.Module):
    """Independent expert module"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Linear(in_features, out_features, bias=False)
        self.weight = self.mlp.weight
        self.weight.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RoleGateWithRoleEmbeddingDualGate(nn.Module):
    """Gating network for role-based expert routing"""
    def __init__(self, input_size: int, num_experts: int, gate_temperature: float = 1.0, role_num: int = 15):
        super().__init__()

        self.lora_role_embedding = nn.Embedding(role_num, input_size)
        # Fix: Linear layer input dimension should be input_size + input_size = 2 * input_size
        self.lora_gate_linear = nn.Linear(input_size, num_experts, bias=False)
        self.lora_gate_role_linear = nn.Linear(input_size, num_experts, bias=False)
        self.lora_gate_softmax = nn.Softmax(dim=-1)  # Apply softmax on the last dimension
        self.gate_temperature = gate_temperature  # Temperature parameter
        
    def forward(self, x: torch.Tensor, role_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_size] or [batch_size, input_size]
            role_id: [batch_size] or [batch_size, seq_len]
        Returns:
            gate_weights: [batch_size, seq_len, num_experts] or [batch_size, num_experts]
        """
        role_embedding = self.lora_role_embedding(role_id).unsqueeze(1)
        
        # Now safe to concat
        # x = torch.cat([x, role_embedding], dim=-1)
        # x = x + role_embedding
        gate_logits = self.lora_gate_linear(x) + self.lora_gate_role_linear(role_embedding)
        # Apply temperature scaling
        scaled_logits = gate_logits / self.gate_temperature
        gate_weights = self.lora_gate_softmax(scaled_logits)
        return gate_weights


class LoraLayer(BaseTunerLayer):
    """
    Role MoE LoRA Layer
    
    Main features:
    1. Supports role-based mixture of experts model (Role MoE)
    2. Manages trainable parameters for experts and gating network
    3. Automatically freezes pretrained weights, ensuring only LoRA parameters are trainable
    4. Provides parameter statistics
    """
    
    logger.info("|> We are using the role_moe_layer with MoE support")
    
    # Base adapter layer names - using a more concise approach
    adapter_layer_names: tuple[str, ...] = (
        "lora_A", 
        "lora_B", 
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_experts_A",  # Expert A matrix - using lora_ prefix
        "lora_experts_B",  # Expert B matrix - using lora_ prefix
        "lora_role_gating",  # Role gating network - using lora_ prefix
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout", "num_experts", "is_moe_adapter", "num_roles")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        
        # Traditional LoRA layers (for non-MoE mode)
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        
        # MoE expert layers
        # Using lora_ prefix ensures parameters won't be frozen by mark_only_lora_as_trainable
        self.lora_experts_A = nn.ModuleDict({})  # {adapter_name: nn.ModuleList([Expert1, Expert2, ...])}
        self.lora_experts_B = nn.ModuleDict({})  # {adapter_name: nn.ModuleList([Expert1, Expert2, ...])}
        
        # Role gating network
        self.lora_role_gating = nn.ModuleDict({})  # {adapter_name: RoleGatingNetwork}
        
        # Role embedding related parameters
        self.use_role_embedding = kwargs.get('use_role_embedding', False)
        
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}  # not actively used anymore after #2443, keep it for BC
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.lora_variant: dict[str, LoraVariant] = {}
        self.kwargs = kwargs
        # Aux loss container for MoE load balancing
        self._aux_loss = None
        
        # Add expert usage statistics related attributes
        self.record_expert_usage = kwargs.get('record_expert_usage', False)
        
        
        # File logging configuration
        self.expert_usage_log_dir = kwargs.get('expert_usage_log_dir', './expert_usage_logs')
        self.expert_usage_log_file = kwargs.get('expert_usage_log_file', None)
        self.layer_name = kwargs.get('layer_name', 'unknown')
        self.gate_weights_history = []
        
        # Ensure log directory exists
        if self.record_expert_usage:
            os.makedirs(self.expert_usage_log_dir, exist_ok=True)
            if self.expert_usage_log_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.expert_usage_log_file = os.path.join(
                    self.expert_usage_log_dir, 
                    f"expert_usage_{timestamp}_{id(self)}.{self.log_format}"
                )

        # Add unlearning related attributes
        self.unlearning_mode = False
        self.forget_roles = []
        self.target_experts = {}  # {role_id: expert_id}
        self._last_gate_weights = None
        self._last_role_ids = None
        self.gating_loss_weight = 0.1  # Gating loss weight

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            torch_supports_dtensor = version.parse(torch.__version__) >= version.parse("2.5.0")
            if torch_supports_dtensor and isinstance(self.base_layer.weight, torch.distributed.tensor.DTensor):
                # If Tensor Parallel is used, the weight is sharded, so we need to get the local shape
                out_features, in_features = self.base_layer.weight.to_local().shape
            else:
                in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "PatchedLinear":
            # INC layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        """Return a matching LoRA variant for this layer type.

        Given the init arguments of this layer, return the correct LoRA variant, if any. E.g., if `use_dora=True`, this
        method should return the DoRA variant for the given layer. If `use_alora=True`, same for aLoRA.

        If there is no fitting variant, return None.

        Note: If this layer type does not support the LoRA variant at all, please raise an error during __init__ as is
        convention, and not here.

        """
        return None

    
    def get_gate_weights_history(self):
        """Get gate_weights history"""
        return self.gate_weights_history

    
    def set_layer_identifier(self, layer_name):
        """Set layer identifier"""
        self.layer_name = layer_name

    def set_record_expert_usage(self, record_expert_usage: bool):
        """Set whether to record expert usage"""
        self.record_expert_usage = record_expert_usage


    def _record_gate_weights(self, gate_weights, experts_features, role_id):
        """Record gate_weights statistics"""
        if not self.record_expert_usage:
            return

        # Sequence-level average
        seq_avg_gate_weights = gate_weights.mean(dim=1)  # [batch_size, num_experts]
        # Convert to float32 to support numpy conversion
        gate_weights_np = seq_avg_gate_weights.detach().cpu().float().numpy()
        role_id_np = role_id.detach().cpu().numpy()
        experts_features = experts_features.mean(dim=1)
        experts_features_np = experts_features.detach().cpu().float().numpy()

        # Handle role_id: could be [batch_size] or [batch_size, seq_len]
        if role_id_np.ndim == 1:
            # [batch_size] - use directly
            role_id_processed = role_id_np.tolist()
        elif role_id_np.ndim == 2:
            # [batch_size, seq_len] - take the first role_id for each batch (or mode)
            # Using mode (most common value) is more reasonable, but for simplicity, take the first here
            role_id_processed = role_id_np[:, 0].tolist()
        else:
            # Other cases, flatten and take the first
            role_id_processed = role_id_np.flatten()[0].item() if role_id_np.size == 1 else role_id_np.flatten()[0]

        # Store to history
        self.gate_weights_history.append({
            'gate_weights': gate_weights_np,
            'role_id': role_id_processed,  # Can now be a single value or list
            'experts_features': experts_features_np,
            'layer_name': self.layer_name,
        })

    # def _record_experts_features(self, )

    def set_unlearning_config(self, forget_roles):
        """
        Set simplified unlearning configuration - only supports negative learning
        
        Args:
            forget_roles: list of role IDs to forget
        """
        self.unlearning_mode = True
        self.forget_roles = forget_roles


    def get_trainable_parameters_count(self) -> dict:
        """Get trainable parameter statistics"""
        stats = {
            'total_trainable': 0,
            'experts_A': 0,
            'experts_B': 0,
            'role_gating': 0,
            'lora_traditional': 0
        }
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats['total_trainable'] += param.numel()
                if 'experts_A' in name:
                    stats['experts_A'] += param.numel()
                elif 'experts_B' in name:
                    stats['experts_B'] += param.numel()
                elif 'role_gating' in name:
                    stats['role_gating'] += param.numel()
                elif 'lora_A' in name or 'lora_B' in name:
                    stats['lora_traditional'] += param.numel()
        
        return stats

    def print_trainable_parameters(self):
        """Print trainable parameter information"""
        stats = self.get_trainable_parameters_count()
        print("=" * 50)
        print("Trainable Parameters Statistics")
        print("=" * 50)
        print(f"Total trainable parameters: {stats['total_trainable']:,}")
        print(f"Experts A parameters: {stats['experts_A']:,}")
        print(f"Experts B parameters: {stats['experts_B']:,}")
        print(f"Role gating parameters: {stats['role_gating']:,}")
        print(f"Traditional LoRA parameters: {stats['lora_traditional']:,}")
        print("=" * 50)

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        use_alora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        arrow_config: ArrowConfig = None,
        qalora_group_size: int = 32,
        inference_mode: bool = False,
        # === New MoE parameters ===
        num_experts: int = None,
        num_roles: int = None,
        use_moe: bool = None,  # Whether to use MoE mode
        gate_temperature: float = 1.0,  # Gating network temperature parameter
        load_balancing_loss_coef: float = 0.0,  # MoE load balancing loss coefficient
        **kwargs,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        if lora_bias and (getattr(self.get_base_layer(), "bias", None) is None):
            warnings.warn(
                f"`lora_bias=True` was passed but the targeted layer of type {type(self.get_base_layer()).__name__} "
                "has no bias. This means that merging LoRA weights won't be possible.",
                PeftWarning,
            )

        lora_variant = self.resolve_lora_variant(
            use_dora=use_dora,
            use_alora=use_alora,
            use_qalora=use_qalora,
            qalora_group_size=qalora_group_size,
            arrow_config=arrow_config,
        )
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # === Store MoE configuration ===
        self.num_roles = num_roles
        self.num_experts = num_experts
        self.use_moe = use_moe  # Whether to use MoE mode
        self.gate_temperature = gate_temperature  # Gating network temperature parameter
        self.load_balancing_loss_coef = load_balancing_loss_coef  # Load balancing loss coefficient
        
        if self.use_moe:
            # Create expert networks
            self.lora_experts_A[adapter_name] = nn.ModuleList([
                Expert(self.in_features, r) for _ in range(self.num_experts)
            ])
            self.lora_experts_B[adapter_name] = nn.ModuleList([
                Expert(r, self.out_features) for _ in range(self.num_experts)
            ])

            self.lora_role_gating[adapter_name] = RoleGateWithRoleEmbeddingDualGate(self.in_features, self.num_experts, gate_temperature=1, role_num=15)
        else:
            # Traditional LoRA mode
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            
            # Ensure traditional LoRA parameters are trainable
            self.lora_A[adapter_name].weight.requires_grad = True
            self.lora_B[adapter_name].weight.requires_grad = True

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.lora_bias[adapter_name] = lora_bias
        self.use_dora[adapter_name] = use_dora

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights == "orthogonal":
            with gather_params_ctx(self.get_base_layer().weight):
                self.orthogonal_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

            self.reset_router_parameters(adapter_name, init_lora_weights)

        # call this before init of the lora variants
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """Override set_adapter method to properly handle MoE modules"""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Call parent class set_adapter method
        super().set_adapter(adapter_names, inference_mode=inference_mode)

        # Check for adapters that were added or removed from the arrow_model.
        # The arrow model may be modified after creation by adding new experts
        # (pre-trained or trainable) or by removing existing ones. Whenever such
        # a change occurs, on_adapter_change() is called to update the set of
        # active task-specific experts and, if needed, to handle recomputing prototypes
        # and doing general knowledge subtraction (GKS) again.
        if hasattr(self, "lora_arrow"):
            for adapter in self.lora_variant:
                if adapter in self.lora_arrow:
                    self.lora_arrow[adapter].on_adapter_change(self.lora_A, self.lora_B)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if self.use_moe and adapter_name in self.lora_experts_A.keys():
            # Initialize MoE experts
            for expert_A, expert_B in zip(self.lora_experts_A[adapter_name], self.lora_experts_B[adapter_name]):
                if init_lora_weights is True:
                    nn.init.kaiming_uniform_(expert_A.weight, a=math.sqrt(5))
                elif init_lora_weights.lower() == "gaussian":
                    nn.init.normal_(expert_A.weight, std=1 / self.r[adapter_name])
                else:
                    raise ValueError(f"Unknown initialization {init_lora_weights=}")
                nn.init.zeros_(expert_B.weight)
                
        elif adapter_name in self.lora_A.keys():
            # Initialize traditional LoRA
            if init_lora_weights is True:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.lora_B[adapter_name].bias)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def reset_router_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        
        if init_lora_weights is True and adapter_name in self.lora_role_gating:
            # Initialize role gating network parameters
            nn.init.kaiming_uniform_(self.lora_role_gating[adapter_name].lora_gate_linear.weight, a=math.sqrt(5))
            if self.lora_role_gating[adapter_name].lora_gate_linear.bias is not None:
                nn.init.zeros_(self.lora_role_gating[adapter_name].lora_gate_linear.bias)
            nn.init.kaiming_uniform_(self.lora_role_gating[adapter_name].lora_gate_role_linear.weight, a=math.sqrt(5))
            if self.lora_role_gating[adapter_name].lora_gate_role_linear.bias is not None:
                nn.init.zeros_(self.lora_role_gating[adapter_name].lora_gate_role_linear.bias)
                #     if role_gate.lora_gate_linear.bias is not None:
                #         nn.init.zeros_(role_gate.lora_gate_linear.bias)

    def olora_init(self, adapter_name):
        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight
        bnb_param_type = get_bnb_param_type(orig_weight)
        dtype = orig_weight.dtype

        if bnb_param_type:
            # check without importing bitsandbytes and robust to bnb_4bit_quant_storage=float*
            weight_tensor = dequantize_module_weight(base_layer)
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = orig_weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        if bnb_param_type == "4bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                quant_type=orig_weight.quant_type,
                quant_storage=orig_weight.quant_storage,
                compress_statistics=orig_weight.compress_statistics,
                module=orig_weight.module,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        elif bnb_param_type == "8bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                requires_grad=orig_weight.requires_grad,
                has_fp16_weights=orig_weight.has_fp16_weights,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        else:
            weight_tensor = weight_tensor.to(dtype)
            base_layer.weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = transpose(weight.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight.data = weight

    def corda_init(self, adapter_name, init_lora_weights):
        linear = self.get_base_layer()
        weight = linear.weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize CorDA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        out_dim = weight.data.size(0)
        in_dim = weight.data.size(1)

        # Calculate WC from covariance matrix
        if not hasattr(linear, "eigens"):
            raise ValueError(
                "`eigens` attribute not found for layer, please run `preprocess_corda` first. "
                "More information can be found at examples/corda_finetuning/README.md."
            )
        eigens = linear.eigens
        U = eigens.U_WC
        S = eigens.S_WC
        V = eigens.V_WC
        r = self.r[adapter_name]

        # nan or inf check
        if torch.isnan(S).any() or torch.isinf(S).any():
            raise ValueError(
                "Invalid value found in matrix S. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if torch.isnan(U).any() or torch.isinf(U).any():
            raise ValueError(
                "Invalid value found in matrix U. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if torch.isnan(V).any() or torch.isinf(V).any():
            raise ValueError(
                "Invalid value found in matrix V. Please file an issue at https://github.com/huggingface/peft/issues."
            )

        # Sanity check
        if U.size(0) != out_dim or U.size(1) != r:
            raise ValueError(
                f"Matrix U size mismatch: {U.size()} vs. ({out_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )
        if S.size(0) != r:
            raise ValueError(
                f"Matrix S size mismatch: {S.size()} vs. ({r},). Please make sure the `lora_config` and `model` argument "
                "of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache in `preprocess_corda`, "
                "please make sure the cache is built with the same model and LoRA rank."
            )
        if V.size(0) != in_dim or V.size(1) != r:
            raise ValueError(
                f"Matrix V size mismatch: {V.size()} vs. ({in_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )

        # Apply alpha
        S /= self.scaling[adapter_name]

        # Init lora_A and lora_B weights
        lora_A = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        lora_B = U.mul(S.sqrt()).contiguous()
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

        # Remove redundant fields
        del linear.eigens

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    @torch.no_grad()
    def orthogonal_init(self, adapter_name):
        # https://datta0.github.io/posts/rethink-lora-init/#orthogonal-initialisation
        rank = self.r[adapter_name]
        if rank % 2 != 0:
            raise ValueError(f"Orthogonal initialization requires the LoRA rank to be even, got {rank} instead.")

        X = torch.randn(rank, rank)
        Q, _ = torch.linalg.qr(X)
        q_odd = Q[0::2, :]  # Odd rows
        q_even = Q[1::2, :]  # Even rows
        dtype = self.get_base_layer().weight.dtype
        lora_A = torch.randn(self.in_features, rank // 2).mm(q_odd).T / 10.0
        lora_B = torch.randn(rank // 2, self.out_features).T.mm(q_even) / 10.0
        self.lora_A[adapter_name].weight = nn.Parameter(lora_A.contiguous().to(dtype))
        self.lora_B[adapter_name].weight = nn.Parameter(lora_B.contiguous().to(dtype))

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter: str, scale: float | int) -> None:
        """Set the scale of the given adapter to the initial scale multiplied by the provided factor

        The initial scale is determined by the configured `r` (rank) and `lora_alpha`.
        """
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float | int) -> None:
        """Multiply the current scale of all active adapters by the provided factor"""
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale: Optional[float | int] = None) -> None:
        """Divide the current scale of all active adapters by the provided factor. If `scale=None` is passed, reset to
        initial scale

        The initial scale is determined by the configured `r` (rank) and `lora_alpha`.

        """
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
        alora_offsets = variant_kwargs.get("alora_offsets", None)
        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            if active_adapter not in self.lora_variant:  # vanilla LoRA
                lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
                result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)
            else:
                if alora_offsets is not None:
                    variant_kwargs["alora_offsets"] = [alora_offsets[j] for j in sub_batch_indices_list[i]]
                lora_output = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=sub_batch,
                    result=result[sub_batch_indices_list[i]],
                    **variant_kwargs,
                    **kwargs,
                )
                result[sub_batch_indices_list[i]] = lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_alora: bool = False,
        arrow_config: ArrowConfig = None,
        lora_bias: bool = False,
        # === New MoE parameters ===
        num_experts: int = None,
        num_roles: int = None,
        use_moe: bool = None,  # Whether to use MoE mode
        gate_temperature: float = 1.0,  # Gating network temperature parameter
        **kwargs,
    ) -> None:
        super().__init__()
        self.role_diversity_loss_coef = kwargs.pop("role_diversity_loss_coef", 0.0)
        self.experts_orthogonal_loss_coef = kwargs.pop("experts_orthogonal_loss_coef", 0.0)
        self.role_diversity_loss_type = kwargs.pop("role_diversity_loss_type", 'kl')
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self.num_roles = num_roles
        self.num_experts = num_experts
        self.use_moe = use_moe
        self.gate_temperature = gate_temperature

        self._role_diversity_loss = None
        self._diversity_penalty = None

        # Freeze pretrained weights, ensuring only LoRA parameters are trainable
        if hasattr(self.base_layer, 'weight'):
            self.base_layer.weight.requires_grad = False
        if hasattr(self.base_layer, 'bias') and self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            use_alora=use_alora,
            lora_bias=lora_bias,
            arrow_config=arrow_config,
            # === Pass MoE parameters ===
            num_experts=num_experts,
            num_roles=num_roles,
            use_moe=use_moe,
            gate_temperature=gate_temperature,
        )

    def resolve_lora_variant(
        self, *, arrow_config: ArrowConfig, use_dora: bool, use_alora: bool, **kwargs
    ) -> Optional[LoraVariant]:
        if arrow_config is not None:
            from .variants import ArrowLinearVariant

            return ArrowLinearVariant()

        if not use_dora and not use_alora:
            return None

        from .variants import ALoraLinearVariant, DoraLinearVariant

        if use_alora:
            return ALoraLinearVariant()
        else:
            return DoraLinearVariant()


    def _compute_role_diversity_penalty(
        self,
        gate_weights: torch.Tensor,   # [B, S, E],
        role_id: torch.Tensor,        # [B, S], role id (int) for each position
        use_js: bool = True,          # True uses JS; False uses symmetric KL
        eps: float = 1e-8,
    ) -> Optional[torch.Tensor]:
        """
        Encourage different roles to use different expert distributions.
        Returns a scalar loss (with negative sign, so minimizing loss = -diversity ⇒ maximizing difference).
        """
        if gate_weights is None or gate_weights.numel() == 0:
            return None
        probs = gate_weights
        if probs.dim() != 3:
            return None

        B, S, E = probs.shape

        # Calculate average expert distribution for each role
        role_vecs = []
        unique_roles = torch.unique(role_id)
        for r in unique_roles.tolist():
            mask = (role_id == r)                     # Shape depends on role_id
            if mask.any():
                
                # Select by batch
                sel = probs[mask]  # [N_true, S, E]
                sel = sel.view(-1, E)  # [N_true*S, E]
               
                role_p = sel.mean(dim=0)              # [E] Get average expert usage distribution for each role
                # Numerical stability protection (optional)
                role_p = torch.clamp(role_p, min=eps) 
                role_p = role_p / role_p.sum()
                role_vecs.append(role_p)

        if len(role_vecs) < 2:
            return probs.new_zeros(())

        role_dists = torch.stack(role_vecs, dim=0)    # [R, E]

        # Calculate pairwise differences
        R = role_dists.size(0)
        total = probs.new_zeros(())
        pairs = 0

        for i in range(R):
            for j in range(i + 1, R):
                p = torch.clamp(role_dists[i], min=eps)
                q = torch.clamp(role_dists[j], min=eps)

                if use_js:
                    m = 0.5 * (p + q)
                    # JS(P,Q) = 0.5*KL(P‖M) + 0.5*KL(Q‖M)
                    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)))
                    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)))
                    d = 0.5 * (kl_pm + kl_qm)
                else:
                    # Symmetric KL
                    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)))
                    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)))
                    d = 0.5 * (kl_pq + kl_qp)

                total = total + d
                pairs += 1

        if pairs == 0:
            return probs.new_zeros(())

        avg_div = total / pairs
        # Penalty term with negative sign: want to "maximize difference", minimize loss during training
        return -avg_div

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_dtype = orig_weight.dtype
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        orig_weight += delta_weight.to(orig_dtype)
                    else:
                        orig_weight = self.lora_variant[active_adapter].merge_safe(self, active_adapter, orig_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight

                    if self.lora_bias[active_adapter]:
                        if getattr(base_layer, "bias", None) is None:
                            raise RuntimeError(
                                "Impossible to merge LoRA with `lora_bias=True` because the base layer has no bias."
                            )
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias * self.scaling[active_adapter]
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias.to(orig_dtype)

                else:
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        base_layer.weight.data += delta_weight
                    else:
                        self.lora_variant[active_adapter].merge_unsafe(self, active_adapter, base_layer.weight)

                    if self.lora_bias[active_adapter]:
                        if getattr(base_layer, "bias", None) is None:
                            raise RuntimeError(
                                "Impossible to merge LoRA with `lora_bias=True` because the base layer has no bias."
                            )
                        base_layer.bias.data += self.lora_B[active_adapter].bias * self.scaling[active_adapter]

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """

        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    orig_dtype = weight.dtype
                    delta_weight = self.get_delta_weight(active_adapter)
                    weight.data -= delta_weight.to(orig_dtype)
                else:
                    unmerged = self.lora_variant[active_adapter].unmerge(self, active_adapter, weight)
                    weight.data = unmerged

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias * self.scaling[active_adapter]

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        if self.use_moe and adapter in self.lora_experts_A.keys():
            # MoE mode: calculate average weight of all experts
            device = self.lora_experts_A[adapter][0].weight.device
            dtype = self.lora_experts_A[adapter][0].weight.dtype
            
            # Collect weights of all experts
            expert_weights_A = []
            expert_weights_B = []
            
            for expert_A, expert_B in zip(self.lora_experts_A[adapter], self.lora_experts_B[adapter]):
                expert_weights_A.append(expert_A.weight)
                expert_weights_B.append(expert_B.weight)
            
            # Calculate average weight (simplified here, may need more complex aggregation strategy in practice)
            avg_weight_A = torch.stack(expert_weights_A).mean(dim=0)
            avg_weight_B = torch.stack(expert_weights_B).mean(dim=0)
            
            # Calculate delta weight
            cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
            
            if cast_to_fp32:
                avg_weight_A = avg_weight_A.float()
                avg_weight_B = avg_weight_B.float()
            
            output_tensor = transpose(avg_weight_B @ avg_weight_A, self.fan_in_fan_out) * self.scaling[adapter]
            
            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)
            
            return output_tensor
            
        elif adapter in self.lora_A.keys():
            # Traditional LoRA mode
            device = self.lora_B[adapter].weight.device
            dtype = self.lora_B[adapter].weight.dtype

            # In case users wants to merge the adapter weights that are in
            # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
            # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
            cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

            weight_A = self.lora_A[adapter].weight
            weight_B = self.lora_B[adapter].weight

            if cast_to_fp32:
                weight_A = weight_A.float()
                weight_B = weight_B.float()

            output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)

                # cast back the weights
                self.lora_A[adapter].weight.data = weight_A.to(dtype)
                self.lora_B[adapter].weight.data = weight_B.to(dtype)

            return output_tensor
        else:
            raise ValueError(f"Adapter {adapter} not found in any weight storage.")

    def forward(self, 
                x: torch.Tensor, 
                role_id: torch.Tensor,
                *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer
        
        # Set whether to record expert usage
        if not self.training:
            self.set_record_expert_usage(True)
        # else:
        #     logger.info("Role Play MoE Layer received no role_id")

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            # === Iterate through all active adapters ===
            for active_adapter in self.active_adapters:
                dropout = self.lora_dropout[active_adapter]
                x_dropped = dropout(x)
                scaling = self.scaling[active_adapter]

                if self.use_moe and active_adapter in self.lora_experts_A.keys():
                    # MoE mode
                    batch_size, seq_len, _ = x_dropped.size()
                    assert role_id is not None, "role_id must be provided for MoE mode"
                    
                    # Use all experts with gating network
                    role_gate = self.lora_role_gating[active_adapter]
                    routing_weights = role_gate(x_dropped, role_id)

                    gate_weights = routing_weights.to(x_dropped.dtype)

                    if self.role_diversity_loss_coef > 0.0:

                        diversity_penalty = self._compute_role_diversity_penalty(gate_weights, role_id)
                        scaled_diversity = self.role_diversity_loss_coef * diversity_penalty
                    else:
                        scaled_diversity = 0.0
                       
                    
                    expert_outputs = []
                    for A_i, B_i in zip(self.lora_experts_A[active_adapter], self.lora_experts_B[active_adapter]):
                        y_i = B_i(A_i(x_dropped))            # [B, S, out_features]
                        expert_outputs.append(y_i)

                    all_outputs = torch.stack(expert_outputs, dim=2) # torch.Size([1, 1024, 3, 4096])
                    lora_output = torch.sum(all_outputs * gate_weights.unsqueeze(-1), dim=2)
                    
                    result = result + lora_output * scaling

                    # all_outputs: [batch, seq_len, num_experts, out_features]
                    if self.experts_orthogonal_loss_coef > 0.0:
                        B, S, E, D = all_outputs.shape

                        # Directly average over sequence dimension to avoid huge reshape operations
                        expert_representations = all_outputs.mean(dim=1)  # [B, E, D]
                        # expert_representations = expert_representations.squeeze(0)  # [E, D]
                        expert_representations = expert_representations.mean(dim=0)
                        expert_representations = torch.nn.functional.normalize(
                            expert_representations, p=2, dim=1, eps=1e-12
                        )

                        # Gram matrix: pairwise inner product of expert outputs
                        G = expert_representations @ expert_representations.T  # [E, E]

                        # Identity matrix as target
                        I = torch.eye(E, device=G.device, dtype=G.dtype)

                        # Orthogonalization loss, only look at upper triangle to avoid duplication
                        orth_loss = torch.triu((G - I) ** 2, diagonal=1).sum() * self.experts_orthogonal_loss_coef
                    else:
                        orth_loss = 0.0
                    # logger.info(f"Orthogonal loss: {orth_loss}")
                    # orth_loss = orth_loss * orthogonal_weight 
                    # PPPAs
                    if not self.training:
                        self._record_gate_weights(gate_weights, result, role_id)

                    # Release memory - clean up in dependency order
                    del expert_outputs  # First clean up list
                    del all_outputs        # Then clean up stacked tensor
                    del lora_output         # Finally clean up computed result
                    # Note: gate_weights may still be used later, so don't delete here
                    
                elif active_adapter in self.lora_A.keys():
                    # Traditional LoRA mode
                    x_casted = x_dropped.to(self.lora_A[active_adapter].weight.dtype)
                    lora_output = self.lora_B[active_adapter](self.lora_A[active_adapter](x_casted)) * scaling
                    result = result + lora_output
                    
                else:
                    # Use LoRA variant
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x_dropped,
                        result=result,
                        **variant_kwargs,
                        **kwargs,
                    )

            result = result.to(torch_result_dtype)

        return result, scaled_diversity, orth_loss

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

