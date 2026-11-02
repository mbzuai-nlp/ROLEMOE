# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig

from loguru import logger


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]


    # @staticmethod
    # def _call_with_role(layer, x, role_id):
    #     # try:
    #     logger.info(f"Calling layer with role_id: {role_id}")
    #     return layer(x, role_id)
    #     # except Exception as e:
            # logger.info(f"Error calling layer with role_id: {role_id}")
            # return layer(x)

    def forward(self, x, role_id=None):
        # Handle gate_proj output (may be tuple with diversity loss)
        if role_id is not None:
            gate_output = self.gate_proj(x, role_id=role_id)
        else:
            gate_output = self.gate_proj(x)
        if isinstance(gate_output, tuple):
            gate, gate_diversity_loss, gate_experts_orthogonal_loss = gate_output
        else:
            gate = gate_output
            gate_diversity_loss = None
            gate_experts_orthogonal_loss = None

        # Handle up_proj output (may be tuple with diversity loss)
        if role_id is not None:
            up_output = self.up_proj(x, role_id=role_id)
        else:
            up_output = self.up_proj(x)
        if isinstance(up_output, tuple):
            up, up_diversity_loss, up_experts_orthogonal_loss = up_output
        else:
            up = up_output
            up_diversity_loss = None
            up_experts_orthogonal_loss = None
            
        hidden = self.act_fn(gate) * up
        
        # Handle down_proj output (may be tuple with diversity loss)
        if role_id is not None:
            down_output = self.down_proj(hidden, role_id=role_id)
        else:
            down_output = self.down_proj(hidden)
        if isinstance(down_output, tuple):
            out, down_diversity_loss, down_experts_orthogonal_loss = down_output
        else:
            out = down_output
            down_diversity_loss = None
            down_experts_orthogonal_loss = None
            
        # Combine diversity losses
        if role_id is not None:
            total_diversity_loss = None
            diversity_losses = [loss for loss in [gate_diversity_loss, up_diversity_loss, down_diversity_loss] if loss is not None]
            if diversity_losses:
                total_diversity_loss = sum(diversity_losses)

            total_experts_orthogonal_loss = None
            experts_orthogonal_losses = [loss for loss in [gate_experts_orthogonal_loss, up_experts_orthogonal_loss, down_experts_orthogonal_loss] if loss is not None]
            if experts_orthogonal_losses:
                total_experts_orthogonal_loss = sum(experts_orthogonal_losses)
        else:
            total_diversity_loss = 0
            total_experts_orthogonal_loss = 0
            
        return out, total_diversity_loss, total_experts_orthogonal_loss


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        role_id: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Capture diversity loss from LoRA layers
        if role_id is not None:
            q_output = self.q_proj(hidden_states, role_id=role_id)
        else:
            q_output = self.q_proj(hidden_states)
        if isinstance(q_output, tuple):
            query_states, q_diversity_loss, q_experts_orthogonal_loss = q_output
            if not hasattr(self, '_diversity_losses'):
                self._diversity_losses = []
            if q_diversity_loss is not None:
                self._diversity_losses.append(q_diversity_loss)

            if not hasattr(self, '_experts_orthogonal_losses'):
                self._experts_orthogonal_losses = []
            if q_experts_orthogonal_loss is not None:
                self._experts_orthogonal_losses.append(q_experts_orthogonal_loss)
            
        else:
            query_states = q_output
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        
        if role_id is not None:
            k_output = self.k_proj(hidden_states, role_id=role_id)
        else:
            k_output = self.k_proj(hidden_states)

        if isinstance(k_output, tuple):
            key_states, k_diversity_loss, k_experts_orthogonal_loss = k_output
            if not hasattr(self, '_diversity_losses'):
                self._diversity_losses = []
            if k_diversity_loss is not None:
                self._diversity_losses.append(k_diversity_loss)
                
            if not hasattr(self, '_experts_orthogonal_losses'):
                self._experts_orthogonal_losses = []
            if k_experts_orthogonal_loss is not None:
                self._experts_orthogonal_losses.append(k_experts_orthogonal_loss)
        else:
            key_states = k_output
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        
        if role_id is not None:
            v_output = self.v_proj(hidden_states, role_id=role_id)
        else:
            v_output = self.v_proj(hidden_states)
            
        if isinstance(v_output, tuple):
            value_states, v_diversity_loss, v_experts_orthogonal_loss = v_output
            if not hasattr(self, '_diversity_losses'):
                self._diversity_losses = []
            if v_diversity_loss is not None:
                self._diversity_losses.append(v_diversity_loss)
                
            if not hasattr(self, '_experts_orthogonal_losses'):
                self._experts_orthogonal_losses = []
            if v_experts_orthogonal_loss is not None:
                self._experts_orthogonal_losses.append(v_experts_orthogonal_loss)
        else:
            value_states = v_output
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        
        # Capture diversity loss from o_proj
        if role_id is not None:
            o_output = self.o_proj(attn_output, role_id=role_id)
        else:
            o_output = self.o_proj(attn_output)
            
        if isinstance(o_output, tuple):
            attn_output, o_diversity_loss, o_experts_orthogonal_loss = o_output
            if not hasattr(self, '_diversity_losses'):
                self._diversity_losses = []
            if o_diversity_loss is not None:
                self._diversity_losses.append(o_diversity_loss)
                
            if not hasattr(self, '_experts_orthogonal_losses'):
                self._experts_orthogonal_losses = []
            if o_experts_orthogonal_loss is not None:
                self._experts_orthogonal_losses.append(o_experts_orthogonal_loss)
        else:
            attn_output = o_output
        
        # Calculate total diversity loss
        total_diversity_loss = None
        if hasattr(self, '_diversity_losses') and self._diversity_losses:
            total_diversity_loss = sum(self._diversity_losses)
            # Clear the list for next forward pass
            self._diversity_losses = []

        total_experts_orthogonal_loss = None
        if hasattr(self, '_experts_orthogonal_losses') and self._experts_orthogonal_losses:
            total_experts_orthogonal_loss = sum(self._experts_orthogonal_losses)
            # Clear the list for next forward pass
            self._experts_orthogonal_losses = []
        
        return attn_output, attn_weights, total_diversity_loss, total_experts_orthogonal_loss


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        role_id: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            role_id=role_id,
            **kwargs,
        )
        if isinstance(attn_output, tuple) and len(attn_output) == 4:
            hidden_states, _, attn_diversity_loss, attn_experts_orthogonal_loss = attn_output
        else:
            hidden_states, _ = attn_output
            attn_diversity_loss = None
            attn_experts_orthogonal_loss = None
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if role_id is not None:
            mlp_output = self.mlp(hidden_states, role_id=role_id)
        else:
            mlp_output = self.mlp(hidden_states)
        if isinstance(mlp_output, tuple):
            hidden_states, mlp_diversity_loss, mlp_experts_orthogonal_loss = mlp_output
        else:
            hidden_states = mlp_output
            mlp_diversity_loss = None
            mlp_experts_orthogonal_loss = None
        hidden_states = residual + hidden_states
        
        # Combine diversity losses
        if attn_diversity_loss is not None and mlp_diversity_loss is not None:
            total_diversity_loss = attn_diversity_loss + mlp_diversity_loss
        elif attn_diversity_loss is not None:
            total_diversity_loss = attn_diversity_loss
        elif mlp_diversity_loss is not None:
            total_diversity_loss = mlp_diversity_loss
        else:
            total_diversity_loss = None
            
        if attn_experts_orthogonal_loss is not None and mlp_experts_orthogonal_loss is not None:
            total_experts_orthogonal_loss = attn_experts_orthogonal_loss + mlp_experts_orthogonal_loss
        elif attn_experts_orthogonal_loss is not None:
            total_experts_orthogonal_loss = attn_experts_orthogonal_loss
        elif mlp_experts_orthogonal_loss is not None:
            total_experts_orthogonal_loss = mlp_experts_orthogonal_loss
        else:
            total_experts_orthogonal_loss = None
        return hidden_states, total_diversity_loss, total_experts_orthogonal_loss


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Role embedding layer - for Role Play MoE LoRA
        # self.num_roles = getattr(config, 'num_roles', 0)
        # if self.num_roles > 0:
        # self.role_embeddings = nn.Embedding(15, config.hidden_size)
        # logger.info(f"Initialized role embeddings for 15 roles")

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        role_id: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        Args:
            role_id (`torch.LongTensor`, *optional*):
                Per-sample role identifiers consumed by role-aware adapters such as Role-Play MoE LoRA. The tensor
                should broadcast over the batch dimension of the input and remains constant across decoded tokens for
                generation.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Prepare role embeddings (if they exist)
        # if role_id is not None and hasattr(self, 'role_embeddings'):
            # role_id: [batch_size] -> [batch_size, hidden_size]
        # role_embedding = self.role_embeddings(role_id)
        # role_embedding = role_embedding.unsqueeze(1)
        # Add role embeddings directly to hidden_states
        # hidden_states = hidden_states + role_embedding  
        diversity_losses = torch.zeros(1)[0].to(hidden_states)
        experts_orthogonal_losses = torch.zeros(1)[0].to(hidden_states)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, diversity_loss, experts_orthogonal_loss = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                role_id=role_id,
                **kwargs,
            )
            if diversity_loss is not None:
                diversity_losses += diversity_loss
            if experts_orthogonal_loss is not None:
                experts_orthogonal_losses += experts_orthogonal_loss

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            diversity_losses=diversity_losses,
            experts_orthogonal_losses=experts_orthogonal_losses)


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        role_id: Optional[torch.LongTensor] = None,
        use_topk_routing: Optional[bool] = None,
        top_k_routing: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Args:
            role_id (`torch.LongTensor`, *optional*):
                Per-example role identifiers used by role-aware adapters (for example, Role-Play MoE LoRA). Supply a
                tensor broadcastable to the batch dimension; the identifier remains the same across generated tokens.
            use_topk_routing (`bool`, *optional*):
                Enable top-k expert routing in role-aware adapters during generation/forward. Passed through to
                adapter layers if supported; otherwise ignored.
            top_k_routing (`int`, *optional*):
                The number of experts to select when `use_topk_routing` is enabled. Passed through to adapter layers
                if supported; otherwise ignored.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```

        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            role_id=role_id,
            use_topk_routing=use_topk_routing,
            top_k_routing=top_k_routing,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            # Print diversity loss
            from loguru import logger
            main_loss_val = loss.item() if loss.numel() == 1 else loss
            self.diversity_losses = outputs.diversity_losses
            self.experts_orthogonal_losses = outputs.experts_orthogonal_losses
            self.main_losses = main_loss_val            
            loss = loss + self.diversity_losses + self.experts_orthogonal_losses

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
