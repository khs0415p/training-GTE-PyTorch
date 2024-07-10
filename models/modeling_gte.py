########### general text embedding
# BASE LM : bert-base-uncased
# learning rate : 2 x 10^4, batch : 16384
# mean pooling
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from .configuration_gte import GTEConfig

def rotate_half(states):
    """
    states : [bs, seq, num_head, head_dim]
    """
    states1 = states[..., : states.shape[-1] // 2] # [bs, seq, num_head, : head_dim//2]
    states2 = states[..., states.shape[-1] // 2 :] # [bs, seq, num_head, head_dim//2 :]
    return torch.cat((-states2, states1), dim=-1) # [bs, seq, num_head, head_dim]


def apply_rotary_pos_emb(query, key, cos, sin):
    cos, sin = cos.to(query.dtype), sin.to(query.dtype)
    query_embed = (query * cos) + (rotate_half(query) * sin)
    key_embed = (key * cos) + (rotate_half(key) * sin)
    return query_embed, key_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=512, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_position_embeddings = seq_len
        t = torch.arange(self.max_position_embeddings, device=device, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # [bs, heads, seq_len, head_size]
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


class NTKScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=512, base=10000, device=None, scaling_factor=1.0, mixed_b=None):
        self.scaling_factor = scaling_factor
        self.mixed_b = mixed_b
        super().__init__(dim, max_position_embeddings, base, device)
        max_position_embeddings = max_position_embeddings * self.scaling_factor
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (self.scaling_factor if self.mixed_b is None else 1)
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

            if self.mixed_b is None:
                inv_freq = inv_freq / self.scaling_factor ** (2 / self.dim)
            else:
                a = torch.tensor(self.scaling_factor).log() / (self.dim / 2) ** self.mixed_b
                lambda_1_m = (a * torch.arange(1, self.dim // 2 + 1).float().to(device) ** self.mixed_b).exp()
                inv_freq = inv_freq / lambda_1_m
            
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class RMSNorm(nn.Module):
    """
    Root Mean Square
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.sqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


LAYER_NORM = {
    'layer_norm': nn.LayerNorm,
    'rms_norm': RMSNorm
}


class Embeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.position_embedding_type = config.position_embedding_type

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )


        if self.position_embedding_type == "rope":
            self._init_rope(config)
        else:
            raise NotImplementedError

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings), persistent=False)

    def _init_rope(self, config):
        kwargs = dict(
            dim=int(config.hidden_size / config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        if config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(**kwargs)
        else:
            kwargs.update(scaling_factor=config.rope_scaling["factor"])
            scaling_type = config.rope_scaling["type"]
            if scaling_type == "ntk":
                kwargs.update(mixed_b=config.rope_scaling.get('mixed_b', None))
                self.rotary_emb = NTKScalingRotaryEmbedding(**kwargs)
            else:
                raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[List[int]]]:
        device, input_shape = input_ids.device, input_ids.shape
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        embeddings = self.word_embeddings(input_ids)

        if position_ids is None:
            if seq_length > self.position_ids.size(0):
                self.register_buffer("position_ids", torch.arange(seq_length), persistent=False)
            position_ids = self.position_ids[:seq_length].expand(batch_size, -1)
        
        if self.position_embedding_type == "rope":
            rope_cos, rope_sin = self.rotary_emb(embeddings, seq_len=seq_length)
            rope_cos = rope_cos[position_ids].unsqueeze(2) # [bs, seq_len, 1, dim]
            rope_sin = rope_sin[position_ids].unsqueeze(2)
            rope_embeds = rope_cos, rope_sin

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = position_ids.mul(0)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings, attention_mask, rope_embeds


class Attention(nn.Module):
    def __init__(self, config, pack_qkv=None):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if pack_qkv is None:
            pack_qkv = config.pack_qkv
        self.pack_qkv = pack_qkv

        if self.pack_qkv:
            self.qkv_proj = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=True)
        else:
            self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
            self.k_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
            self.v_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.FloatTensor,
        rope_embeds: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        qkv_inputs: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, ...]:
        shape_hd = (self.num_attention_heads, self.attention_head_size)

        if self.pack_qkv and qkv_inputs is None:
            qkv_pack = self.qkv_proj(hidden_states).split(self.all_head_size, dim=-1)
        else:
            if qkv_inputs is None:
                qkv_inputs = (hidden_states, hidden_states, hidden_states)
            qkv_pack = [
                getattr(self, n + '_proj')(s) for s, n in zip(qkv_inputs, 'qkv')
            ]
        query_states, key_states, value_states = [t.view(t.shape[:-1] + shape_hd) for t in qkv_pack] # [bs, seq, num_head, head_dim]

        if self.config.position_embedding_type == "rope":
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *rope_embeds)

        context_layer = self._attention(query_states, key_states, value_states, attention_bias, head_mask)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [bs, seq, dim]
        context_layer = context_layer.view(new_context_layer_shape)

        attn_output = self.o_proj(context_layer)

        outputs = (attn_output,)
        return outputs

    def _attention(self, query_states, key_states, value_states, attention_bias, head_mask):
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        if attention_bias is not None:
            attention_scores = attention_scores + attention_bias

        attention_probs = F.softmax(attention_scores, dim=-1)
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores = self.dropout(attention_scores)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [bs, seq, num_head, head_dim]
        return context_layer


class GatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.up_gate_proj = nn.Linear(config.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]
        if config.hidden_dropout_prob > 0:
            self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            self.hidden_dropout = None

    def forward(self, hidden_states):
        up_gate = self.up_gate_proj(hidden_states)
        up_states, gate = torch.split(up_gate, self.intermediate_size, dim=-1)
        gate = self.act_fn(gate)
        gated_states = gate * up_states
        if self.hidden_dropout is not None:
            gated_states = self.hidden_dropout(gated_states)
        down_states = self.down_proj(gated_states)
        return down_states


class Layer(nn.Module):
    def __init__(self, config, pack_qkv=None):
        super().__init__()
        self.attention = Attention(config, pack_qkv)
        self.mlp = GatedMLP(config)

        ln_class = LAYER_NORM[config.layer_norm_type]
        self.attn_ln = ln_class(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_ln = ln_class(config.hidden_size, eps=config.layer_norm_eps)

        if config.hidden_dropout_prob > 0:
            self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            self.hidden_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.FloatTensor,
        rope_embeds: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        qkv_inputs: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states
        attention_outputs = self.attention(
            hidden_states,
            attention_bias,
            rope_embeds,
            head_mask,
            qkv_inputs,
        )
        hidden_states = attention_outputs[0]
        if self.hidden_dropout is not None:
            hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.attn_ln(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        if self.hidden_dropout is not None:
            hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.mlp_ln(hidden_states)

        outputs = (hidden_states,)
        return outputs
    

class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        rope_embeds: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_bias,
                rope_embeds,
                layer_head_mask,
            )

            hidden_states = layer_outputs[0]
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states]
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GTEPretrainedModel(PreTrainedModel):
    config_class = GTEConfig
    base_model_prefix = "gte"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GTEModel(GTEPretrainedModel):
    def __init__(self, config: GTEConfig, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

        self.pooler = Pooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        else:
            raise ValueError

        (embedding_output, attention_mask, rope_embeds) = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        batch_size, seq_length = input_shape

        attention_bias = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_bias=attention_bias,
            rope_embeds=rope_embeds,
            head_mask=head_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        seqeunce_output = encoder_outputs[0]
        pooled_output = self.pooler(seqeunce_output) if self.pooler is not None else None

        if not return_dict:
            return (seqeunce_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=seqeunce_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class GTELMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states