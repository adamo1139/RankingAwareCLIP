from typing import Optional, Callable, Literal
from dataclasses import dataclass

import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import CLIP
from .coca_model_adapter import CrossModalQKVFusion
from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    AttentionalPooler,
)


@dataclass
class AdapterCfg:
    num_layers: int = 2
    d_model: int = 768
    n_head: int = 8
    num_learnable_text_tokens: int = 0
    adapter_attention_pooler: Literal[
        'softmax',
        'sigmoid',
        'rank-attn',
    ] = 'rank-attn'
    cross_attn_method: Literal['cross-qkv'] = 'cross-qkv'
    reg_token_method: Literal['cross-attn-sum'] = 'concat'
    n_reg_tokens: int = 16


def _build_rank_adapter(
    adapter_cfg: AdapterCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
    model_version: Optional[str] = None,
):
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )
    if model_version == 'clip-adapter-v2':
        adapter = RankAdapterV2(
            num_layers=adapter_cfg.num_layers,
            d_model=adapter_cfg.d_model,
            n_head=adapter_cfg.n_head,
            cross_attn_method=adapter_cfg.cross_attn_method,
            num_learnable_text_tokens=adapter_cfg.num_learnable_text_tokens,
            adapter_attention_pooler=adapter_cfg.adapter_attention_pooler,
            reg_token_method=adapter_cfg.reg_token_method,
            act_layer=act_layer,
            norm_layer=norm_layer,
            n_reg_tokens=adapter_cfg.n_reg_tokens,
        )
    else:
        raise NotImplementedError(f'{model_version} not implemented.')
    return adapter


class L2RCLIP(CLIP):
    def __init__(
        self,
        adapter_cfg,
        misc_cfg,
        quick_gelu: bool = False,
        model_version: Optional[str] = None,
        cast_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            **kwargs,
        )
        adapter_cfg = AdapterCfg(**adapter_cfg)
        self.adapter = _build_rank_adapter(
            adapter_cfg=adapter_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            model_version=misc_cfg['model_version'],
        )

    def lock_text_tower(self, unlocked_layers, freeze_layer_norm):
        print("<<< Force Lock Text Tower >>>")
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(
        self,
        image,
        text: Optional[torch.Tensor] = None,
        adapter_targets: Optional[torch.Tensor] = None,
        is_training=True,
    ):
        image_features = self.encode_image(
            image,
            # normalize=True,
        )  # (N, d_model, h, w)
        text_features, text_embs = self.encode_text(
            text,
            output_tokens=True,
            # normalize=True,
        )  # 1:(N, ctx_t, d_model)

        # print(f'DEBUG: {image_features.shape=}, {text_features.shape=}')
        image_embs = image_features.clone()
        if len(image_embs.shape) == 4:
            # CNN-backbone (ViT: B, n_ctx, D -- no need additional operation)
            bz, d_model, im_h, im_w = image_features.shape
            image_embs = image_embs.view(bz, d_model, -1).permute(0, 2, 1)  # (N, ctx_v, d_model)
        adapter_logits, adapter_paired_logits = self.adapter(image_embs, text_embs)  # (N, 1)

        out_dict = {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': self.logit_scale.exp(),
            'adapter_logits': adapter_logits,
            'adapter_targets': adapter_targets,
            'adapter_paired_logits': adapter_paired_logits,
        }
        if self.logit_bias is not None:
            out_dict['logit_bais'] = self.logit_bias
        return out_dict


class RankAdapterV2(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_head: int,
        cross_attn_method: str,
        num_learnable_text_tokens: int,
        adapter_attention_pooler: str,
        reg_token_method: str,
        n_reg_tokens: int,
        act_layer: Callable,
        norm_layer: Callable,
    ):
        super().__init__()
        self.cross_attn_method = cross_attn_method
        self.reg_token_method = reg_token_method
        self.n_reg_tokens = n_reg_tokens
        self.adapter_attention_pooler = adapter_attention_pooler

        if num_learnable_text_tokens > 0:
            scale = num_learnable_text_tokens ** -0.5
            self.learnable_text_tokens = nn.Parameter(
                scale * torch.randn((1, num_learnable_text_tokens))
            )
        else:
            self.learnable_text_tokens = None

        if cross_attn_method == 'null':
            print('Disable Cross-Modal Fusion')
            self.crossmodal_block = nn.Identity()

        if cross_attn_method == 'cross-qkv':
            self.crossmodal_block = CrossModalQKVFusion(
                width=d_model,
                heads=8,
                layers=num_layers,
                ls_init_value=None,
                output_dim=d_model,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            raise NotImplementedError

        if self.reg_token_method == 'null':
            print('Disable Regression Token Fusion (but keep reg_token params)')
            self.post_fuse_block = nn.Identity()
        else:
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=0.1,
                batch_first=True,
            )
            self.post_fuse_block = nn.TransformerEncoder(
                encoder_layers,
                num_layers=num_layers,
            )

        if n_reg_tokens > 0:
            if adapter_attention_pooler in ['softmax', 'sigmoid']:
                self.reg_token_pooler = AttentionalPooler(
                    d_model=d_model,
                    context_dim=d_model,
                    n_queries=n_reg_tokens,
                    use_distributed_attn=adapter_attention_pooler == 'sigmoid',
                    distributed_attn_ops=adapter_attention_pooler,
                )
            elif adapter_attention_pooler == 'rank-attn':
                scale = d_model ** -0.5
                self.reg_tokens = nn.Parameter(
                    scale * torch.randn(self.n_reg_tokens, d_model),
                )
                self.reg_token_pooler = MultiHeadRankawareAttentionV2(
                    d_model=d_model,
                    norm_layer=norm_layer,
                    num_heads=8,
                )
            elif adapter_attention_pooler == 'null':
                print(f'No Ranking-aware attention')
                self.reg_token_pooler = nn.Identity()
            else:
                raise NotImplementedError(f'{adapter_attention_pooler} not implemented.')
        else:
            print('#REG tokens=0. Use Identity and sum over all patch-tokens')
            self.reg_token_pooler = nn.Identity()

        self.pair_regressor = PairwiseDiffBlockV2(
            d_model,
            norm_layer,
        )
        self.reg_ln = norm_layer(d_model)
        self.regressor = MLPLayer(
            d_model,
            norm_layer,
        )

    def forward(
        self,
        image_embeddings,
        text_embeddings,
    ):
        batch_size, num_visual_token, d_model = image_embeddings.shape
        device = image_embeddings.device

        if self.learnable_text_tokens is not None:
            learnable_text_tokens = self.learnable_text_tokens.view(1, -1, 1).expand(
                batch_size, -1, text_embeddings.size(-1)
            )
            text_embeddings = torch.cat(
                (
                    learnable_text_tokens.to(device),
                    text_embeddings,
                ),
                dim=1,
            )
        x = self.crossmodal_block(
            image_embeddings,
            text_embeddings,
        )
        x = self.post_fuse_block(x)  # [N, L, D]

        if self.adapter_attention_pooler in ['rank-attn']:
            reg_tokens = self.reg_tokens
            reg_tokens = reg_tokens.unsqueeze(0)
            pair_pos_embedding = None
            x_pair_diff = self.reg_token_pooler(
                reg_tokens,  # query
                x,  # key/value,
                pair_pos_embedding,
            )
            x_pair_diff = x_pair_diff.sum(axis=1)
            x_pair_diff = self.pair_regressor(x_pair_diff)
            x = self.reg_ln(x)
            x = self.regressor(x.sum(axis=1))
        elif self.adapter_attention_pooler == 'null':
            x_pair_diff = (x.unsqueeze(1) - x.unsqueeze(0))
            x_pair_diff = x_pair_diff.view(batch_size ** 2, num_visual_token, d_model)
            x_pair_diff = x_pair_diff.mean(axis=1)  # apply a averging op over tokens
            x_pair_diff = self.pair_regressor(x_pair_diff)

            # REG part
            x = self.reg_ln(x)
            x = self.regressor(x.sum(axis=1))
        else:
            x = self.reg_token_pooler(x)  # [N, L', D]
            x = x.sum(axis=1)  # [N, D], # Sum before compute diff
            x_pair_diff = self.pair_regressor(x)
            x = self.regressor(x)

        return x, x_pair_diff


class PairwiseDiffBlockV2(nn.Module):
    """Used for Attn-Rank.
    """
    def __init__(
        self,
        input_dim: int,
        norm_layer: Callable,
    ):
        super().__init__()

        # Make transform, prevent using the same embedding with regression head
        self.pre_linear_block = nn.Sequential(*nn.ModuleList([
            # norm the input (if not normed)
            norm_layer(input_dim),
            # sub-network
            nn.Linear(input_dim, input_dim),
            norm_layer(input_dim),
            nn.ReLU(),
            ##
            nn.Linear(input_dim, input_dim),
            norm_layer(input_dim),
            nn.ReLU(),
            ##
            nn.Linear(input_dim, input_dim),
            norm_layer(input_dim),
            nn.ReLU(),
        ]))  # input has been normed, use Linear -> norm -> act

        self.output_head = nn.Linear(input_dim, 1)

        nn.init.xavier_normal_(self.pre_linear_block[1].weight)
        nn.init.xavier_normal_(self.pre_linear_block[4].weight)
        nn.init.xavier_normal_(self.pre_linear_block[7].weight)
        nn.init.xavier_normal_(self.output_head.weight)

    def forward(self, x):
        x = self.pre_linear_block(x)
        x = self.output_head(x)
        return x


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        norm_layer: Callable,
    ):
        super().__init__()
        self.pre_linear_block = nn.Sequential(*nn.ModuleList([
            nn.Linear(input_dim, input_dim),
            norm_layer(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            norm_layer(input_dim),
            nn.ReLU(),
        ]))  # input has been normed, use Linear -> norm -> act
        self.output_head = nn.Linear(
            input_dim,
            1,
        )

        nn.init.xavier_normal_(self.pre_linear_block[0].weight)
        nn.init.xavier_normal_(self.pre_linear_block[3].weight)
        nn.init.xavier_normal_(self.output_head.weight)

    def forward(self, x):
        x = self.pre_linear_block(x)
        x = self.output_head(x)
        return x


class RankawareAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        norm_layer: Callable,
    ):
        super().__init__()
        self.ln_q = norm_layer(d_model)
        self.ln_kv = norm_layer(d_model)

    def forward(
        self,
        q_x: torch.Tensor,  # [1, n.Q, D]
        kv_x: torch.Tensor,  # [N, n.P, D]
        return_attention: bool = False,
    ):
        _, n_query_tokens, latent_dim = q_x.shape
        batch_size, n_patch_tokens, _ = kv_x.shape
        q_x = q_x.expand(batch_size ** 2, -1, -1)

        q_x = self.ln_q(q_x)
        kv_x = self.ln_kv(kv_x)
        kv_x = (kv_x.unsqueeze(1) - kv_x.unsqueeze(0)).view(-1, n_patch_tokens, latent_dim)

        mat1 = q_x @ kv_x.mT  # batch matmul
        mat1 = mat1 / math.sqrt(latent_dim)  # [N, n.Q, n.P]
        attention = F.softmax(mat1, dim=-1)

        values = attention @ kv_x
        if return_attention:
            return values, attention
        return values


class MultiHeadRankawareAttentionV2(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        norm_layer: Callable,
        kv_use_bn: bool = False,
        ablation_mode: Optional[str] = None,  # To support extended ablation study
    ):
        super().__init__()
        self.num_heads = num_heads
        self.kv_use_bn = kv_use_bn
        self.d_head = d_model // num_heads  # Dimension per head
        self.scale = self.d_head ** -0.5  # Scaling factor for dot-product attention
        self.ablation_mode = ablation_mode

        self.ln_q = norm_layer(d_model)
        if kv_use_bn:
            self.ln_kv = nn.BatchNorm1d(d_model)
        else:
            self.ln_kv = norm_layer(d_model)

        # Linear layers to project into query, key, and value for each head
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model)

        # Final projection layer to combine the heads' outputs
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q_x: torch.Tensor,  # [1, n.Q, D]
        kv_x: torch.Tensor,  # [N, n.P, D]
        pair_pos_embedding=None,
        return_attention: bool = False,
    ):
        _, n_query_tokens, d_model = q_x.shape
        batch_size, n_patch_tokens, _ = kv_x.shape

        # Apply normalization and project into multi-head space
        q_x = self.ln_q(self.q_proj(q_x)).view(
            1, n_query_tokens, self.num_heads, self.d_head
        ).transpose(1, 2)  # [1, num_heads, n.Q, d_head]
        if self.kv_use_bn:
            kv_x = self.kv_proj(kv_x)
            kv_x = self.ln_kv(kv_x.permute(0, 2, 1))
            kv_x = kv_x.permute(0, 2, 1).view(
                batch_size, n_patch_tokens, self.num_heads, self.d_head
            ).transpose(1, 2)
        else:
            kv_x = self.ln_kv(self.kv_proj(kv_x)).view(
                batch_size, n_patch_tokens, self.num_heads, self.d_head,
            ).transpose(1, 2)  # [N, num_heads, n.P, d_head]

        # Expand query to match batch size (both dimensions need to match)
        q_x = q_x.expand(batch_size ** 2, self.num_heads, n_query_tokens, self.d_head)

        # Concatenate kv_x for multi-head attention
        kv_x_repeated = kv_x.repeat_interleave(batch_size, dim=0)  # Repeat for batch size [N**2, num_heads, n.P, d_head]
        kv_x_tiled = kv_x.tile(batch_size, 1, 1, 1)  # Tile for batch size [N**2, num_heads, n.P, d_head]

        # Concatenate along the patch dimension (dim=2)
        kv_x_concat = torch.cat([
            kv_x_repeated,
            kv_x_tiled,
        ], dim=2)  # [N**2, num_heads, 2 * n.P, d_head]

        if pair_pos_embedding is not None:
            pair_pos_embedding = pair_pos_embedding.view(1, 1, 2 * n_patch_tokens, d_model)
            pair_pos_embedding = pair_pos_embedding.permute(
                0, 2, 1, 3
            ).view(
                1, 2 * n_patch_tokens, self.num_heads, d_model // self.num_heads,
            ).permute(0, 2, 1, 3)  # Prevent reshape to token embedding
            pair_pos_embedding = pair_pos_embedding.expand(
                batch_size ** 2,
                self.num_heads,
                2 * n_patch_tokens,
                d_model // self.num_heads,
            )
            kv_x_concat += pair_pos_embedding

        mat1 = torch.einsum(
            'bnqd,bnpd->bnqp',
            q_x, kv_x_concat,
        ) * self.scale  # [N**2, num_heads, n.Q, 2 * n.P]

        attention = F.softmax(mat1, dim=-1)
        attn_1 = attention[..., :n_patch_tokens]
        attn_2 = attention[..., n_patch_tokens:]

        values_1 = torch.einsum(
            'bnqp,bnpd->bnqd',
            attn_1,
            kv_x_concat[:, :, :n_patch_tokens, :],
        )  # [N**2, num_heads, n.Q, d_head]
        values_2 = torch.einsum(
            'bnqp,bnpd->bnqd',
            attn_2,
            kv_x_concat[:, :, n_patch_tokens:, :],
        )  # [N**2, num_heads, n.Q, d_head]

        values = values_1 - values_2  # [N**2, num_heads, n.Q, d_head]
        values = values.transpose(1, 2).contiguous().view(
            batch_size ** 2,
            n_query_tokens,
            d_model,
        )  # [N**2, n.Q, D]

        values = self.out_proj(values)

        if return_attention:
            return values, attention
        return values