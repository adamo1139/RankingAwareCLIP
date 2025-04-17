from typing import Optional, Callable, Literal, Union
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .coca_model import CoCa
from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    AttentionalPooler,
    Transformer,
    ResidualAttentionBlock,
)


@dataclass
class AdapterCfg:
    num_layers: int = 2
    d_model: int = 512
    n_head: int = 8
    has_batch_attention: Union[bool, str] = False  # True/False, iterative-qkv
    num_learnable_text_tokens: Optional[int] = None
    with_batch_index_coding: bool = False
    token_pooling_for_batch_block: bool = False
    adapter_after: Literal['unimodal-decoder', 'multimodal-decoder'] = 'unimodal-decoder'
    cross_attn_method: Literal['fuse-self-attn', 'cross-qkv-attn', 'null'] = 'fuse-self-attn'
    output_pooler: Literal['adaptive-pooling', 'attn-pooler'] = 'adaptive-pooling'
    with_paired_emb_branch: bool = False


ADAPTER_CLS = 'v5'
VISUAL_LETENT_FLAG = 'patch'  # from v5: patch, otherwise latent
FORWARD_VISUAL_POS = True  # True: v5b; False: v5
# maybe change it later (to version that force copy)
SIGMOID_POOLER = True  # only When LINEAR_POOLER is False (False: Use original)

LINEAR_POOLER = False  # (No more this opt)True for v5d (but its performance ...)


def _build_rank_adapter(
    adapter_cfg: AdapterCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )
    """
    no-prefix: result -- v3
    V2: result --v4a
    V3 -- v4b
    V4 -- v4c
    """
    adapter_ver = {
        'v4a': RankAdapterV2,
        'v4b': RankAdapterV3,
        'v4c': RankAdapterV4,
        'v5': RankAdapterV5,
    }
    adapter_cls = adapter_ver[ADAPTER_CLS]
    adapter = adapter_cls(
        num_layers=adapter_cfg.num_layers,
        d_model=adapter_cfg.d_model,
        n_head=adapter_cfg.n_head,
        has_batch_attention=adapter_cfg.has_batch_attention,
        num_learnable_text_tokens=adapter_cfg.num_learnable_text_tokens,
        with_batch_index_coding=adapter_cfg.with_batch_index_coding,
        adapter_after=adapter_cfg.adapter_after,
        token_pooling_for_batch_block=adapter_cfg.token_pooling_for_batch_block,
        act_layer=act_layer,
        norm_layer=norm_layer,
        cross_attn_method=adapter_cfg.cross_attn_method,
        output_pooler=adapter_cfg.output_pooler,
        with_paired_emb_branch=adapter_cfg.with_paired_emb_branch,
    )
    return adapter


class L2RCoCa(CoCa):
    def __init__(
        self,
        adapter_cfg,
        quick_gelu: bool = False,
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
        )

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        print('Lock Visual Tower')
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers=0, freeze_layer_norm=True):
        print('Lock Text Encoder (No unlock allow, keep args for competibility)')
        for param in self.text.parameters():
            param.requires_grad = False

    def forward(
        self,
        image,
        text: Optional[torch.Tensor] = None,
        adapter_targets: Optional[torch.Tensor] = None,
        image_latent: Optional[torch.Tensor] = None,
        image_embs: Optional[torch.Tensor] = None,
        is_training=True,
    ):
        if image_latent is None or image_embs is None:
            if VISUAL_LETENT_FLAG == 'latent':
                # [N, n_dim], [N, ctx_v, n_dim]
                image_latent, image_embs = self._encode_image(image)
            elif VISUAL_LETENT_FLAG == 'patch':
                image_latent, image_embs, patch_tok_embs = self._encode_image(
                    image,
                    return_patch_tokens=True,
                )

        if text is None:
            return {
                "image_features": image_latent,
                "image_embs": image_embs
            }

        # [N, n_dim], [N, ctx_t, n_dim]
        text_latent, token_embs = self._encode_text(text)

        # TODO: add assertion to avoid bugs? (GitLab)
        labels = text[:, 1:]  # skip <start-of-sentence>, SOS
        if is_training:
            token_embs = token_embs[:, :-1]  # skip <end-of-sentence>, EOS
        logits, decoder_token_embs = self.text_decoder(
            image_embs,
            token_embs,
            return_x_before_project=True,
        )  # [N, ctx_t - 1, vocab_size], [N, ctx_t - 1, n_dim]
        # NOTE: If forward to adapter, use generated one or this one

        adapter_logits, adapter_paired_logits = self.adapter(
            image_embs if VISUAL_LETENT_FLAG == 'latent' else patch_tok_embs,
            token_embs,
            self.visual.positional_embedding if FORWARD_VISUAL_POS else None,
        )  # [N, 1]

        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "adapter_logits": adapter_logits,
            "adapter_targets": adapter_targets,
            "adapter_paired_logits": adapter_paired_logits,
            "logit_scale": self.logit_scale.exp()
        }
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict


class RankAdapter(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_head: int,
        has_batch_attention: Union[bool, str],
        num_learnable_text_tokens: Optional[int],
        with_batch_index_coding: bool,
        token_pooling_for_batch_block: bool,
        adapter_after: Literal['unimodal-decoder', 'multimodal-decoder'],
        act_layer: Callable,
        norm_layer: Callable,
        cross_attn_method: Literal['fuse-self-attn', 'cross-qkv-attn', 'null'],
        output_pooler: Literal['adaptive-pooling', 'attn-pooler']
    ):
        super().__init__()
        self.token_pooling_for_batch_block = token_pooling_for_batch_block
        self.cross_attn_method = cross_attn_method
        self.has_batch_attention = has_batch_attention
        self.output_pooler = output_pooler

        # log-info
        print(f'{num_learnable_text_tokens=}, {cross_attn_method=}')
        print(f'{has_batch_attention=}, {with_batch_index_coding=}, {token_pooling_for_batch_block=}')
        print(f'{output_pooler=}')

        if (num_learnable_text_tokens is not None) and (num_learnable_text_tokens > 0):
            print(f'Model has learnable text tokens: {num_learnable_text_tokens}')
            self.learnable_text_tokens = nn.Parameter(
                torch.randn((1, num_learnable_text_tokens))
            )
            nn.init.normal_(self.learnable_text_tokens, std=0.02)
        else:
            self.learnable_text_tokens = None

        if with_batch_index_coding:
            print('Model has learnable batch-index encoding')
            max_batch_size = 64  # TODO: make this configurable
            self.batch_index_embedding = nn.Parameter(
                torch.randn((max_batch_size, d_model)),
            )
            nn.init.normal_(self.batch_index_embedding, std=0.02)
        else:
            self.batch_index_embedding = None

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model,
            dropout=0.1,
            batch_first=True,
            # activation=act_layer,
        )
        if cross_attn_method == 'fuse-self-attn':
            self.crossmodal_block = nn.TransformerEncoder(
                encoder_layers,
                num_layers=num_layers,
            )
        elif cross_attn_method == 'cross-qkv-attn':
            self.crossmodal_block = CrossModalQKVFusion(
                width=d_model,
                heads=8,
                layers=num_layers,
                ls_init_value=None,
                output_dim=d_model,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        elif cross_attn_method == 'null':
            self.crossmodal_block = nn.Identity()
        else:
            raise NotImplementedError

        if has_batch_attention and has_batch_attention in ['iterative-qkv', 'iterative-qkv-first']:
            # backward competible
            self.batch_attention_block = CrossImageQKVFusion(
                width=d_model,
                # width=64,
                heads=8,
                layers=num_layers,
                ls_init_value=None,
                output_dim=d_model,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        elif has_batch_attention:
            batch_encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=0.1,
                batch_first=True,
            )
            self.batch_attention_block = nn.TransformerEncoder(
                batch_encoder_layers,
                num_layers=num_layers,
            )
        else:
            self.batch_attention_block = nn.Identity()
        self.ca_post_ln = norm_layer(d_model)
        self.ba_post_ln = norm_layer(d_model)

        if output_pooler == 'adaptive-pooling':
            self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        elif output_pooler == 'attn-pooler':
            self.pooling_layer = AttentionalPooler(
                d_model,
                d_model,
                n_queries=1,
            )
            self.pooled_post_ln = norm_layer(d_model)
        else:
            raise NotImplementedError

        self.mlp_block = MLPLayer(
            in_features_dim=d_model,
            out_features_dim=1,
            dropout_ratio=0.,
        )

    def forward(self, image_embeddings, text_embeddings, **kwargs):
        """
        image_embeddings: [N, v_ctx, ndim]
        text_embeddings: [N, t_ctx, ndim]
        """
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device

        if self.learnable_text_tokens is not None:
            learnable_text_tokens = self.learnable_text_tokens.view(1, -1, 1).expand(
                batch_size, -1, text_embeddings.size(-1)
            )
            text_embeddings = torch.cat((
                learnable_text_tokens.to(device),
                text_embeddings,
            ), dim=1)

        if self.cross_attn_method == 'fuse-self-attn':
            x = torch.cat([
                image_embeddings,
                text_embeddings,
            ], dim=1)  # [N, v_ctx + t_ctx, ndim]
            x = self.crossmodal_block(x)  # [N, v_ctx + t_ctx, ndim]
            x = self.ca_post_ln(x)
        elif self.cross_attn_method == 'cross-qkv-attn':
            x = self.crossmodal_block(
                image_embeddings,
                text_embeddings,
            )
            # norm has neen processed inside the module
        elif self.cross_attn_method == 'null':
            # no use text info (or shall we just add/mul them?)
            x = self.crossmodal_block(image_embeddings)
        else:
            raise NotImplementedError

        # TODO: Think about batch index embedding as class-token
        # Or it can use pooled one as Q, previous output as K,V (attention mode)
        # batch-interact
        if self.token_pooling_for_batch_block:
            pooled_ = torch.cat([
                x.min(dim=1, keepdim=True)[0],
                x.mean(dim=1, keepdim=True),
                x.max(dim=1, keepdim=True)[0],
            ], dim=1)  # [N, 3, ndim]
            if self.has_batch_attention == 'iterative-qkv':
                raise NotImplementedError
            else:
                pooled_ = pooled_.permute(1, 0, 2)
                pooled_ = self.batch_attention_block(pooled_)
                pooled_ = pooled_.permute(1, 0, 2)
            self.ba_post_ln(pooled_)
            x = torch.cat([x, pooled_], dim=1)
        else:
            if self.has_batch_attention == 'iterative-qkv':
                _, token_dim, hidden_dim = x.shape
                outs = []
                for batch_index in range(batch_size):
                    q_index = batch_index
                    # --- #
                    # NOTE: v20240215 implementation: this only tuple, not attend to whole batch
                    # kv_index = (batch_index + 1) % batch_size
                    # q_embedding = x[q_index].view(-1, token_dim, hidden_dim)
                    # kv_embedding = x[kv_index].view(-1, token_dim, hidden_dim)
                    # x_ = self.batch_attention_block(
                    #     q_embedding,
                    #     kv_embedding,
                    # )
                    # --- #
                    # NOTE: lead to OOM? -- bz8 ~ 8k, bz16 ~ 16k, bz32 might have 32k
                    q_embedding = x[q_index].view(-1, token_dim, hidden_dim).expand(
                        (batch_size, token_dim, hidden_dim)
                    )
                    x_ = self.batch_attention_block(
                        q_embedding,
                        x,  # include self
                    )
                    x_ = x_.mean(dim=0, keepdim=True)
                    # --- #
                    outs.append(x_)
                x = torch.concatenate(outs)
            elif self.has_batch_attention == 'iterative-qkv-first':
                _, token_dim, hidden_dim = x.shape
                q_embedding = x[0].view(-1, token_dim, hidden_dim).expand(
                    (batch_size, token_dim, hidden_dim)
                )
                x = self.batch_attention_block(
                    q_embedding,
                    x,
                )
            else:
                x = x.permute(1, 0, 2)
                x = self.batch_attention_block(x)
                x = x.permute(1, 0, 2)
            self.ba_post_ln(x)

        if self.output_pooler == 'adaptive-pooling':
            x = self.pooling_layer(x.permute(0, 2, 1)).squeeze(-1)
        else:
            x = self.pooling_layer(x).squeeze(1)
            x = self.pooled_post_ln(x)
        x = self.mlp_block(x)  # [N, 1]
        # return F.sigmoid(x) * 10  # logit-scale  # might not good, since the loss will take sigmoid
        # print(f'{x=}')
        return F.softplus(x)


class RankAdapterV2(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_head: int,
        has_batch_attention: Union[bool, str],
        num_learnable_text_tokens: Optional[int],
        with_batch_index_coding: bool,
        token_pooling_for_batch_block: bool,
        adapter_after: Literal['unimodal-decoder', 'multimodal-decoder'],
        act_layer: Callable,
        norm_layer: Callable,
        cross_attn_method: Literal['fuse-self-attn', 'cross-qkv-attn', 'null'],
        output_pooler: Literal['adaptive-pooling', 'attn-pooler', 'null'],  # TODO: may not useful, keep to make the code competible
        with_paired_emb_branch: bool = False,
    ):
        super().__init__()
        self.token_pooling_for_batch_block = token_pooling_for_batch_block
        self.cross_attn_method = cross_attn_method
        self.has_batch_attention = has_batch_attention
        self.with_batch_index_coding = with_batch_index_coding
        self.with_paired_emb_branch = with_paired_emb_branch

        # log-info
        print(f'{num_learnable_text_tokens=}, {cross_attn_method=}')
        print(f'{has_batch_attention=}, {with_batch_index_coding=}')
        print('Version2')
        #
        print('<<< Not used functions')
        print(f'{output_pooler=}')
        print(f'{token_pooling_for_batch_block=}')
        print('Not used functions >>>')

        if (num_learnable_text_tokens is not None) and (num_learnable_text_tokens > 0):
            print(f'Model has learnable text tokens: {num_learnable_text_tokens}')
            self.learnable_text_tokens = nn.Parameter(
                torch.randn((1, num_learnable_text_tokens))
            )
            nn.init.normal_(self.learnable_text_tokens, std=0.02)
        else:
            self.learnable_text_tokens = None

        max_batch_size = 64
        self.batch_reg_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.batch_reg_token, std=0.02)
        # always build it but not necessary for forwarding
        self.batch_index_encoding = nn.Parameter(
            torch.randn((max_batch_size, 1, d_model)),
        )
        nn.init.normal_(self.batch_index_encoding, std=0.02)

        if cross_attn_method == 'fuse-self-attn':
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=0.1,
                batch_first=True,
                # activation=act_layer,
            )
            self.crossmodal_block = nn.TransformerEncoder(
                encoder_layers,
                num_layers=num_layers,
            )
        elif cross_attn_method == 'cross-qkv-attn':
            self.crossmodal_block = CrossModalQKVFusion(
                width=d_model,
                heads=8,
                layers=num_layers,
                ls_init_value=None,
                output_dim=d_model,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        elif cross_attn_method == 'null':
            self.crossmodal_block = nn.Identity()
        else:
            raise NotImplementedError

        if has_batch_attention:
            batch_encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=0.1,
                batch_first=True,
            )
            self.batch_attention_block = nn.TransformerEncoder(
                batch_encoder_layers,
                num_layers=num_layers,
            )
        else:
            self.batch_attention_block = nn.Identity()
        self.ca_post_ln = norm_layer(d_model)
        self.ba_post_ln = norm_layer(d_model)

        if with_paired_emb_branch:
            self.paired_branch_mlp1 = MLPLayer(
                in_features_dim=d_model,
                out_features_dim=d_model // 2,
            )
            self.paired_branch_mlp2 = MLPLayer(
                in_features_dim=d_model // 2,
                out_features_dim=d_model // 2,
            )
            # FORM2 (additional layers, no mean pooling)
            # self.p_mlp1_norm = norm_layer(d_model // 2)
            # self.p_mlp2_norm = norm_layer(d_model // 2)
            # self.paired_branch_out = MLPLayer(
            #     in_features_dim=d_model // 2,
            #     out_features_dim=1,
            # )

        # Maybe not useful
        # self.pre_norm_image = norm_layer(d_model)
        # self.pre_norm_text = norm_layer(d_model)

        # self.mlp_block = MLPLayer(
        #     in_features_dim=d_model,
        #     out_features_dim=1,
        #     dropout_ratio=0.1,
        # )
        self.mlp1 = MLPLayer(
            in_features_dim=d_model,
            out_features_dim=d_model,
            dropout_ratio=0.,
        )
        self.mlp2 = MLPLayer(
            in_features_dim=d_model,
            out_features_dim=1,
            dropout_ratio=0.,
        )

    def forward(self, image_embeddings, text_embeddings, **kwargs):
        """
        image_embeddings: [N, v_ctx, ndim]
        text_embeddings: [N, t_ctx, ndim]
        """
        batch_size, _, d_model = image_embeddings.shape
        device = image_embeddings.device

        # Normalize the difference sources embeddings first
        # image_embeddings = self.pre_norm_image(image_embeddings)
        # text_embeddings = self.pre_norm_text(text_embeddings)

        if self.learnable_text_tokens is not None:
            learnable_text_tokens = self.learnable_text_tokens.view(1, -1, 1).expand(
                batch_size, -1, text_embeddings.size(-1)
            )
            text_embeddings = torch.cat((
                learnable_text_tokens.to(device),
                text_embeddings,
            ), dim=1)
        # always concat the reg_token to the front of the image_embedding (like cls_token)
        batch_reg_token = self.batch_reg_token.expand((batch_size, 1, -1))
        if self.with_batch_index_coding:
            batch_reg_token = batch_reg_token + self.batch_index_encoding[:batch_size]
        image_embeddings = torch.concatenate([
            batch_reg_token.to(device),
            image_embeddings,
        ], dim=1)

        if self.cross_attn_method == 'fuse-self-attn':
            x = torch.cat([
                image_embeddings,
                text_embeddings,
            ], dim=1)  # [N, v_ctx + t_ctx, ndim]
            x = self.crossmodal_block(x)  # [N, v_ctx + t_ctx, ndim]
            x = self.ca_post_ln(x)
        elif self.cross_attn_method == 'cross-qkv-attn':
            x = self.crossmodal_block(
                image_embeddings,
                text_embeddings,
            )
            # norm has neen processed inside the module
        elif self.cross_attn_method == 'null':
            # no use text info (or shall we just add/mul them?)
            x = self.crossmodal_block(image_embeddings)
        else:
            raise NotImplementedError

        xx = None  # for competibility
        if not self.has_batch_attention:
            # If no ba, no passing though BA block (or it is identify...)
            x = x[:, 0]  # [N, d_model]
        elif self.has_batch_attention == 'all-emb':
            # Pass all embedding through block and take first for final
            x = x.permute(1, 0, 2)
            x = self.batch_attention_block(x)
            x = self.ba_post_ln(x)
            x = x.permute(1, 0, 2)
            x = x[:, 0]
        elif (self.has_batch_attention == 'single-emb') or self.has_batch_attention:
            # Use single token and pass-through BA block
            # Default if True, if there are more condition, add it before this cond.
            tok_emb = x[:, 0].unsqueeze(0)  # [N, 1, d_model] -> [1, N, d_model]
            x = self.batch_attention_block(tok_emb).squeeze(0)
            x = self.ba_post_ln(x)

            # [EXPERIMENTAL: pairwise distance]
            if self.with_paired_emb_branch:
                # [FORM1: w/ SmoothL1 for pairwise distance]
                xx = self.paired_branch_mlp1(tok_emb.squeeze(0))
                xx = self.paired_branch_mlp2(xx)  # [N, d]
                xx = xx[:, None] - xx[None, :]  # [N, N, d]  # FORM1/2
                xx = xx.mean(axis=-1).view(-1, 1)  # [N*N, 1]

                # [FORM1A: w/ TripletLoss, neg/pos/anchor defined in loss]
                # xx = self.paired_branch_mlp1(tok_emb.squeeze(0))
                # xx = self.paired_branch_mlp2(xx)  # [N, d]

                # [FORM2: w/ SmoothL1 for pairwise, no pooling but with learnable linear]
                # xx = self.paired_branch_mlp1(tok_emb.squeeze(0))
                # xx = self.p_mlp1_norm(xx)
                # xx = self.paired_branch_mlp2(xx)  # [N, d]
                # xx = self.p_mlp2_norm(xx)
                # xx = xx[:, None] - xx[None, :]  # [N, N, d]  # FORM1/2
                # xx = xx.view(-1, xx.size(-1))
                # xx = self.paired_branch_out(xx)
        else:
            raise RuntimeError  # should not triggered.

        x = self.mlp2(self.mlp1(x))
        return x, xx


class RankAdapterV3(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_head: int,
        has_batch_attention: Union[bool, str],
        num_learnable_text_tokens: Optional[int],
        with_batch_index_coding: bool,
        token_pooling_for_batch_block: bool,
        adapter_after: Literal['unimodal-decoder', 'multimodal-decoder'],
        act_layer: Callable,
        norm_layer: Callable,
        cross_attn_method: Literal['fuse-self-attn', 'cross-qkv-attn', 'null'],
        output_pooler: Literal['adaptive-pooling', 'attn-pooler', 'null'],  # TODO: may not useful, keep to make the code competible
        with_paired_emb_branch: bool = False,
    ):
        super().__init__()
        self.token_pooling_for_batch_block = token_pooling_for_batch_block
        self.cross_attn_method = cross_attn_method
        self.has_batch_attention = has_batch_attention
        self.with_batch_index_coding = with_batch_index_coding
        self.with_paired_emb_branch = with_paired_emb_branch

        # log-info
        print('Version3')
        print(f'{num_learnable_text_tokens=}, {cross_attn_method=}')
        print(f'{has_batch_attention=}, {with_batch_index_coding=}')
        #
        print('<<< Not used functions')
        print(f'{output_pooler=}')
        print(f'{token_pooling_for_batch_block=}')
        print(f'{with_paired_emb_branch=}')
        print('Not used functions >>>')

        if (num_learnable_text_tokens is not None) and (num_learnable_text_tokens > 0):
            print(f'Model has learnable text tokens: {num_learnable_text_tokens}')
            self.learnable_text_tokens = nn.Parameter(
                torch.randn((1, num_learnable_text_tokens))
            )
            nn.init.normal_(self.learnable_text_tokens, std=0.02)
        else:
            self.learnable_text_tokens = None

        max_batch_size = 64
        self.batch_reg_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.batch_reg_token, std=0.02)
        # always build it but not necessary for forwarding
        self.batch_index_encoding = nn.Parameter(
            torch.randn((max_batch_size, 1, d_model)),
        )
        nn.init.normal_(self.batch_index_encoding, std=0.02)

        if cross_attn_method == 'fuse-self-attn':
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=0.1,
                batch_first=True,
                # activation=act_layer,
            )
            self.crossmodal_block = nn.TransformerEncoder(
                encoder_layers,
                num_layers=num_layers,
            )
        elif cross_attn_method == 'cross-qkv-attn':
            self.crossmodal_block = CrossModalQKVFusion(
                width=d_model,
                heads=8,
                layers=num_layers,
                ls_init_value=None,
                output_dim=d_model,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        elif cross_attn_method == 'null':
            self.crossmodal_block = nn.Identity()
        else:
            raise NotImplementedError

        if has_batch_attention:
            batch_encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=0.1,
                batch_first=True,
            )
            self.batch_attention_block = nn.TransformerEncoder(
                batch_encoder_layers,
                num_layers=num_layers,
            )
        else:
            self.batch_attention_block = nn.Identity()
        self.ca_post_ln = norm_layer(d_model)
        self.ba_post_ln = norm_layer(d_model)

        if with_paired_emb_branch:
            self.paired_branch_mlp1 = MLPLayer(
                in_features_dim=d_model,
                out_features_dim=d_model // 2,
            )
            self.paired_branch_mlp2 = MLPLayer(
                in_features_dim=d_model // 2,
                out_features_dim=d_model // 2,
            )
            # FORM2 (additional layers, no mean pooling)
            # self.p_mlp1_norm = norm_layer(d_model // 2)
            # self.p_mlp2_norm = norm_layer(d_model // 2)
            # self.paired_branch_out = MLPLayer(
            #     in_features_dim=d_model // 2,
            #     out_features_dim=1,
            # )

        self.reg_token_fusion_block = CrossModalQKVFusion(
            width=512,
            heads=8,
            layers=num_layers,
            ls_init_value=None,
            output_dim=d_model,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.mlp1 = MLPLayer(
            in_features_dim=d_model,
            out_features_dim=d_model,
            dropout_ratio=0.,
        )
        self.mlp2 = MLPLayer(
            in_features_dim=d_model,
            out_features_dim=1,
            dropout_ratio=0.,
        )

    def forward(self, image_embeddings, text_embeddings, **kwargs):
        """
        image_embeddings: [N, v_ctx, ndim]
        text_embeddings: [N, t_ctx, ndim]
        """
        batch_size, _, d_model = image_embeddings.shape
        device = image_embeddings.device

        # Normalize the difference sources embeddings first
        # image_embeddings = self.pre_norm_image(image_embeddings)
        # text_embeddings = self.pre_norm_text(text_embeddings)

        if self.learnable_text_tokens is not None:
            learnable_text_tokens = self.learnable_text_tokens.view(1, -1, 1).expand(
                batch_size, -1, text_embeddings.size(-1)
            )
            text_embeddings = torch.cat((
                learnable_text_tokens.to(device),
                text_embeddings,
            ), dim=1)
        # always concat the reg_token to the front of the image_embedding (like cls_token)
        batch_reg_token = self.batch_reg_token.expand((batch_size, 1, -1))
        if self.with_batch_index_coding:
            batch_reg_token = batch_reg_token + self.batch_index_encoding[:batch_size]

        if self.cross_attn_method == 'fuse-self-attn':
            x = torch.cat([
                image_embeddings,
                text_embeddings,
            ], dim=1)  # [N, v_ctx + t_ctx, ndim]
            x = self.crossmodal_block(x)  # [N, v_ctx + t_ctx, ndim]
            x = self.ca_post_ln(x)
        elif self.cross_attn_method == 'cross-qkv-attn':
            x = self.crossmodal_block(
                image_embeddings,
                text_embeddings,
            )
            # norm has neen processed inside the module
        elif self.cross_attn_method == 'null':
            # no use text info (or shall we just add/mul them?)
            x = self.crossmodal_block(image_embeddings)
        else:
            raise NotImplementedError

        # Post regression token attention
        x = self.reg_token_fusion_block(
            batch_reg_token,  # token as query
            x,  # fused-embedding as k/v
        )  # [N, num_token, d], num_token = 1 for now

        xx = None  # for competibility
        if not self.has_batch_attention:
            # If no ba, no passing though BA block (or it is identify...)
            x = x[:, 0]  # [N, d_model]
        elif self.has_batch_attention == 'all-emb':
            # Pass all embedding through block and take first for final
            x = x.permute(1, 0, 2)
            x = self.batch_attention_block(x)
            x = self.ba_post_ln(x)
            x = x.permute(1, 0, 2)
            x = x[:, 0]
        elif (self.has_batch_attention == 'single-emb') or self.has_batch_attention:
            # Use single token and pass-through BA block
            # Default if True, if there are more condition, add it before this cond.
            tok_emb = x[:, 0].unsqueeze(0)  # [N, 1, d_model] -> [1, N, d_model]
            x = self.batch_attention_block(tok_emb).squeeze(0)
            x = self.ba_post_ln(x)

            # [EXPERIMENTAL: pairwise distance]
            if self.with_paired_emb_branch:
                # [FORM1: w/ SmoothL1 for pairwise distance]
                xx = self.paired_branch_mlp1(tok_emb.squeeze(0))
                xx = self.paired_branch_mlp2(xx)  # [N, d]
                xx = xx[:, None] - xx[None, :]  # [N, N, d]  # FORM1/2
                xx = xx.mean(axis=-1).view(-1, 1)  # [N*N, 1]

                # [FORM1A: w/ TripletLoss, neg/pos/anchor defined in loss]
                # xx = self.paired_branch_mlp1(tok_emb.squeeze(0))
                # xx = self.paired_branch_mlp2(xx)  # [N, d]

                # [FORM2: w/ SmoothL1 for pairwise, no pooling but with learnable linear]
                # xx = self.paired_branch_mlp1(tok_emb.squeeze(0))
                # xx = self.p_mlp1_norm(xx)
                # xx = self.paired_branch_mlp2(xx)  # [N, d]
                # xx = self.p_mlp2_norm(xx)
                # xx = xx[:, None] - xx[None, :]  # [N, N, d]  # FORM1/2
                # xx = xx.view(-1, xx.size(-1))
                # xx = self.paired_branch_out(xx)
        else:
            raise RuntimeError  # should not triggered.

        x = self.mlp2(self.mlp1(x))
        return x, xx


class RankAdapterV4(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_head: int,
        has_batch_attention: Union[bool, str],
        num_learnable_text_tokens: Optional[int],
        with_batch_index_coding: bool,
        token_pooling_for_batch_block: bool,
        adapter_after: Literal['unimodal-decoder', 'multimodal-decoder'],
        act_layer: Callable,
        norm_layer: Callable,
        cross_attn_method: Literal['fuse-self-attn', 'cross-qkv-attn', 'null'],
        output_pooler: Literal['adaptive-pooling', 'attn-pooler', 'null'],  # TODO: may not useful, keep to make the code competible
        with_paired_emb_branch: bool = False,
    ):
        super().__init__()
        self.token_pooling_for_batch_block = token_pooling_for_batch_block
        self.cross_attn_method = cross_attn_method
        self.has_batch_attention = has_batch_attention
        self.with_batch_index_coding = with_batch_index_coding
        self.with_paired_emb_branch = with_paired_emb_branch
        self.num_layers = num_layers

        # log-info
        print('Version4')
        print(f'{num_learnable_text_tokens=}, {cross_attn_method=}')
        print(f'{has_batch_attention=}, {with_batch_index_coding=}')
        #
        print('<<< Not used functions')
        print(f'{output_pooler=}')
        print(f'{token_pooling_for_batch_block=}')
        print(f'{with_paired_emb_branch=}')
        print('Not used functions >>>')

        if (num_learnable_text_tokens is not None) and (num_learnable_text_tokens > 0):
            print(f'Model has learnable text tokens: {num_learnable_text_tokens}')
            self.learnable_text_tokens = nn.Parameter(
                torch.randn((1, num_learnable_text_tokens))
            )
            nn.init.normal_(self.learnable_text_tokens, std=0.02)
        else:
            self.learnable_text_tokens = None

        max_batch_size = 64
        self.batch_reg_token = nn.Parameter(
            torch.randn(1, num_layers, d_model)
        )
        nn.init.normal_(self.batch_reg_token, std=0.02)
        # always build it but not necessary for forwarding
        self.batch_index_encoding = nn.Parameter(
            torch.randn((max_batch_size, 1, d_model)),
        )
        nn.init.normal_(self.batch_index_encoding, std=0.02)

        if cross_attn_method == 'cross-qkv-attn':
            crossmodal_block_list = []
            reg_block_list = []
            for _ in range(num_layers):
                crossmodal_block = CrossModalQKVFusion(
                    width=d_model,
                    heads=8,
                    layers=1,
                    ls_init_value=None,
                    output_dim=d_model,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                crossmodal_block_list.append(
                    crossmodal_block
                )

                reg_token_fusion_block = CrossModalQKVFusion(
                    width=512,
                    heads=8,
                    layers=1,
                    ls_init_value=None,
                    output_dim=d_model,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                reg_block_list.append(
                    reg_token_fusion_block
                )
            self.crossmodal_block_list = nn.ModuleList(
                crossmodal_block_list
            )
            self.reg_block_list = nn.ModuleList(
                reg_block_list
            )
        else:
            print('Version4 only accept I/T cross')
            raise NotImplementedError

        self.reg_token_pooler = AttentionalPooler(
            d_model=d_model,
            context_dim=d_model,
            n_queries=1,
        )  # Use to integrate reg-tokens from different level

        if has_batch_attention:
            batch_encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model,
                dropout=0.1,
                batch_first=True,
            )
            self.batch_attention_block = nn.TransformerEncoder(
                batch_encoder_layers,
                num_layers=num_layers,
            )
        else:
            self.batch_attention_block = nn.Identity()

        if with_paired_emb_branch:
            self.paired_branch_mlp1 = MLPLayer(
                in_features_dim=d_model,
                out_features_dim=d_model // 2,
            )
            self.paired_branch_mlp2 = MLPLayer(
                in_features_dim=d_model // 2,
                out_features_dim=d_model // 2,
            )

        self.mlp1 = MLPLayer(
            in_features_dim=d_model,
            out_features_dim=d_model,
            dropout_ratio=0.,
        )
        self.mlp2 = MLPLayer(
            in_features_dim=d_model,
            out_features_dim=1,
            dropout_ratio=0.,
        )

    def forward(self, image_embeddings, text_embeddings, **kwargs):
        """
        image_embeddings: [N, v_ctx, ndim]
        text_embeddings: [N, t_ctx, ndim]
        """
        batch_size, _, d_model = image_embeddings.shape
        device = image_embeddings.device

        if self.learnable_text_tokens is not None:
            learnable_text_tokens = self.learnable_text_tokens.view(1, -1, 1).expand(
                batch_size, -1, text_embeddings.size(-1)
            )
            text_embeddings = torch.cat((
                learnable_text_tokens.to(device),
                text_embeddings,
            ), dim=1)
        # always concat the reg_token to the front of the image_embedding (like cls_token)
        batch_reg_token = self.batch_reg_token.expand(
            (batch_size, self.num_layers, -1)
        )
        if self.with_batch_index_coding:
            batch_reg_token = batch_reg_token + self.batch_index_encoding[:batch_size]

        if self.cross_attn_method == 'cross-qkv-attn':
            xrs = []
            for i, (cm_fusion_layer, reg_fusion_layer) in enumerate(zip(self.crossmodal_block_list, self.reg_block_list)):
                if i == 0:
                    x = cm_fusion_layer(
                        image_embeddings,
                        text_embeddings,
                    )
                else:
                    x = cm_fusion_layer(
                        x,
                        text_embeddings,
                    )
                xr = reg_fusion_layer(
                    batch_reg_token[:, i, :].unsqueeze(1),
                    x,
                )
                xrs.append(xr)
            xrs = torch.concat(xrs, dim=1)
        else:
            raise NotImplementedError
        x = self.reg_token_pooler(xrs)

        if not self.has_batch_attention:
            # If no ba, no passing though BA block (or it is identify...)
            x = x[:, 0]  # [N, d_model]
        elif (self.has_batch_attention == 'single-emb') or self.has_batch_attention:
            # Use single token and pass-through BA block
            # Default if True, if there are more condition, add it before this cond.
            tok_emb = x[:, 0].unsqueeze(0)  # [N, 1, d_model] -> [1, N, d_model]
            x = self.batch_attention_block(tok_emb).squeeze(0)
            x = self.ba_post_ln(x)

            # [EXPERIMENTAL: pairwise distance]
            if self.with_paired_emb_branch:
                # [FORM1: w/ SmoothL1 for pairwise distance]
                xx = self.paired_branch_mlp1(tok_emb.squeeze(0))
                xx = self.paired_branch_mlp2(xx)  # [N, d]
                xx = xx[:, None] - xx[None, :]  # [N, N, d]  # FORM1/2
                xx = xx.mean(axis=-1).view(-1, 1)  # [N*N, 1]
        else:
            raise RuntimeError  # should not triggered.

        x = self.mlp2(self.mlp1(x))
        return x, None


class RankAdapterV5(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_head: int,
        has_batch_attention: Union[bool, str],
        num_learnable_text_tokens: Optional[int],
        with_batch_index_coding: bool,
        token_pooling_for_batch_block: bool,
        adapter_after: Literal['unimodal-decoder', 'multimodal-decoder'],
        act_layer: Callable,
        norm_layer: Callable,
        cross_attn_method: Literal['fuse-self-attn', 'cross-qkv-attn', 'null'],
        output_pooler: Literal['adaptive-pooling', 'attn-pooler', 'null'],  # TODO: may not useful, keep to make the code competible
        with_paired_emb_branch: bool = False,
    ):
        super().__init__()
        self.token_pooling_for_batch_block = token_pooling_for_batch_block
        self.cross_attn_method = cross_attn_method
        self.has_batch_attention = has_batch_attention
        self.with_batch_index_coding = with_batch_index_coding
        self.with_paired_emb_branch = with_paired_emb_branch
        self.num_layers = num_layers

        # log-info
        print('Version5')
        print(f'{num_learnable_text_tokens=}, {cross_attn_method=}')
        print(f'{has_batch_attention=}, {with_batch_index_coding=}')
        #
        print('<<< Not used functions')
        print(f'{output_pooler=}')
        print(f'{token_pooling_for_batch_block=}')
        print(f'{with_paired_emb_branch=}')
        print('Not used functions >>>')

        if (num_learnable_text_tokens is not None) and (num_learnable_text_tokens > 0):
            print(f'Model has learnable text tokens: {num_learnable_text_tokens}')
            scale = num_learnable_text_tokens ** -0.5
            self.learnable_text_tokens = nn.Parameter(
                scale * torch.randn((1, num_learnable_text_tokens))
            )
        else:
            self.learnable_text_tokens = None

        if cross_attn_method == 'cross-qkv-attn':
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
            print('Version5 only accept I/T cross')
            raise NotImplementedError

        self.patch_token_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.ReLU(inplace=True),
        )

        original_d = 768 if FORWARD_VISUAL_POS else 512
        scale = original_d ** -0.5
        self.reg_tokens = nn.Parameter(
            scale * torch.randn(50, original_d)
        )
        if FORWARD_VISUAL_POS:
            self.visual_positional_emb_proj = nn.Parameter(
                scale * torch.randn(original_d, d_model)
            )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model,
            dropout=0.1,
            batch_first=True,
            # activation=act_layer,
        )
        self.post_fuse_block = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers,
        )
        self.reg_fuse_post_ln = norm_layer(d_model)  # maybe not use

        if not LINEAR_POOLER:
            self.reg_token_pooler = AttentionalPooler(
                d_model=d_model,
                context_dim=d_model,
                n_queries=1,
                use_distributed_attn=SIGMOID_POOLER,
            )  # Use to integrate reg-tokens from different level
        else:
            # need activation? --> No, here is a linear comb only
            self.reg_token_pooler = nn.Linear(
                in_features=49,  # n-tokens ... dirty way
                out_features=1,
                bias=False,
            )

        # Exp: Output standard
        # self.ln_final = norm_layer(d_model)
        # self.mlp = MLPLayer(
        #     in_features_dim=d_model,
        #     out_features_dim=1,
        #     dropout_ratio=0.,
        # )

        # Exp: Output simple NN
        self.mlp = nn.Linear(
            in_features=d_model,
            out_features=1,
        )

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        v_positional_encoding=None,
    ):
        """
        image_embeddings: [N, v_ctx, ndim]
        text_embeddings: [N, t_ctx, ndim]
        """
        image_embeddings = self.patch_token_proj(
            image_embeddings,
        )
        batch_size, _, d_model = image_embeddings.shape
        device = image_embeddings.device

        if self.learnable_text_tokens is not None:
            learnable_text_tokens = self.learnable_text_tokens.view(1, -1, 1).expand(
                batch_size, -1, text_embeddings.size(-1)
            )
            text_embeddings = torch.cat((
                learnable_text_tokens.to(device),
                text_embeddings,
            ), dim=1)

        if self.cross_attn_method == 'cross-qkv-attn':
            x = self.crossmodal_block(
                image_embeddings,  # query
                text_embeddings,  # key/value
            )
        else:
            raise NotImplementedError

        reg_tokens = self.reg_tokens
        if v_positional_encoding is not None:
            reg_tokens = reg_tokens + v_positional_encoding
            reg_tokens = reg_tokens @ self.visual_positional_emb_proj

        reg_tokens = reg_tokens.expand(
            (batch_size, 50, d_model),
        )
        x = torch.concatenate([x, reg_tokens], dim=1)
        x = self.post_fuse_block(x)

        # only take newly added for linear-comb
        if not LINEAR_POOLER:
            x = x[:, 51:]  # (N, m, d)
            # Standard, use a attentional pooler
            x = self.reg_token_pooler(x)  # (N, 1, d)

            # Alternative: either max/avg pool showed inferior performance
            # x = x.sum(axis=1)

            x = x.view(batch_size, -1)

        else:
            x = x[:, 51:]  # whatever, only take patch-token...
            x = x.view(batch_size, d_model, -1)  # (N, d, m)
            # use linear or just avg-pool? (linear)
            x = self.reg_token_pooler(x)  # (N, d, 1)
            x = x.view(batch_size, -1)  # (N, d)

        # OutputExp: Standard: MLP
        # x = self.ln_final(x)
        # x = self.mlp(x)

        # OutputExp: | 1xmlp
        x = self.mlp(x)

        # OutputExp: Simple sum > Failed
        # x = x.sum(axis=-1, keepdims=True)

        return x, None


class MLPLayer(nn.Module):
    def __init__(
        self,
        in_features_dim: int,
        hidden_features_dim: Optional[int] = None,
        out_features_dim: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        dropout_ratio: float = 0.,
    ):
        super().__init__()
        hidden_features_dim = hidden_features_dim or in_features_dim
        out_features_dim = out_features_dim or in_features_dim
        self.fc1 = nn.Linear(in_features_dim, hidden_features_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features_dim, out_features_dim)
        self.dropout = nn.Dropout(dropout_ratio)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CrossModalQKVFusion(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])
        self.ln_final = norm_layer(width)
        self.projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.projection is not None:
            nn.init.normal_(
                self.projection,
                std=self.transformer.width ** -0.5,
            )

    def forward(self, image_embs, text_embs, return_attention=False):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND

        all_attn_matrices = []
        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            image_embs = resblock(image_embs)  # No need masking, let it see all
            if return_attention:
                image_embs, attn_matrix = cross_attn(
                    image_embs,
                    k_x=text_embs,
                    v_x=text_embs,
                    return_attention=True,
                )
                all_attn_matrices.append(attn_matrix)
            else:
                image_embs = cross_attn(
                    image_embs,
                    k_x=text_embs,
                    v_x=text_embs,
                )
        x = image_embs.permute(1, 0, 2)
        x = self.ln_final(x)

        if self.projection is not None:
            x = x @ self.projection

        if return_attention:
            return x, all_attn_matrices
        return x


class CrossImageQKVFusion(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=False,  # TODO: make it False, true apply an additional layer norm to kv
            )
            for _ in range(layers)
        ])
        self.ln_final = norm_layer(width)
        self.projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.projection is not None:
            nn.init.normal_(
                self.projection,
                std=self.transformer.width ** -0.5,
            )

    def forward(self, q_embeddings, kv_embeddings):
        q_embeddings = q_embeddings.permute(1, 0, 2)  # NLD -> LNDsq
        kv_embeddings = kv_embeddings.permute(1, 0, 2)  # NLD -> LND

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            q_embeddings = resblock(q_embeddings)  # No need masking, let it see all
            q_embeddings = cross_attn(
                q_embeddings,
                k_x=kv_embeddings,
                v_x=kv_embeddings,
            )
        x = q_embeddings.permute(1, 0, 2)
        x = self.ln_final(x)

        if self.projection is not None:
            x = x @ self.projection
        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
