# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
import logging

import numpy as np
import torch
from torch import nn, Tensor, FloatTensor
import torch.nn.functional as F
from .modules import (
    Fp32LayerNorm,
    MultiheadAttention,
    SamePad,
    init_bert_params,
    get_activation_fn,
    TransposeLast,
    GLU_Linear,
)

logger = logging.getLogger(__name__)


def pad_for_wavlm(wave: Tensor) -> Tensor:
    """Same padding with frame centering.

    Args:
        wave :: (..., T=t)     - Waveform
    Returns:
             :: (..., T=l+t+r) - Padded waveform
    """

    kernel, hop = 400, 320
    padding_total = kernel - hop
    padding_l = padding_total // 2
    padding_r = padding_total - padding_l
    padded_wave = F.pad(wave, (padding_l, padding_r), mode="constant")

    return padded_wave


class WavLMConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"     # mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with normalize=True)
        self.encoder_layers: int = 12     # num encoder layers in the transformer

        self.encoder_embed_dim: int = 768     # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072     # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12     # num encoder attention heads
        self.activation_fn: str = "gelu"     # activation function to use

        self.layer_norm_first: bool = False     # apply layernorm first in the transformer
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"     # string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_bias: bool = False     # include bias in conv encoder
        self.feature_grad_mult: float = 1.0     # multiply feature extractor var grads by this

        self.normalize: bool = False  # normalize input to have 0 mean and unit variance during training

        # dropouts
        self.dropout: float = 0.1     # dropout probability for the transformer
        self.attention_dropout: float = 0.1     # dropout probability for attention weights
        self.activation_dropout: float = 0.0     # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0     # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0     # dropout to apply to the input (after feat extr)
        self.dropout_features: float = 0.0     # dropout to apply to the features (after feat extr)

        # masking
        self.mask_length: int = 10     # mask length
        self.mask_prob: float = 0.65     # probability of replacing a token with mask
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # channel masking
        self.mask_channel_length: int = 10     # length of the mask for features (channels)
        self.mask_channel_prob: float = 0.0     # probability of replacing a feature with 0
        self.mask_channel_selection: str = "static"     # how to choose mask length for channel masking
        self.mask_channel_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
        self.no_mask_channel_overlap: bool = False     # whether to allow channel masks to overlap
        self.mask_channel_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # positional embeddings
        self.conv_pos: int = 128     # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16     # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False     # apply relative position embedding
        self.num_buckets: int = 320     # number of buckets for relative position embedding
        self.max_distance: int = 1280     # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False     # apply gated relative position embedding

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class WavLM(nn.Module):
    """WavLM model. Output match is confirmed."""

    def __init__(self, cfg: WavLMConfig) -> None:
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        self.cfg = cfg
        feat_feature = 512
        feat_enc_i = 1024

        self.feature_extractor = ConvFeatureExtractionModel()
        self.layer_norm        = nn.LayerNorm(feat_feature)
        self.post_extract_proj = nn.Linear(feat_feature, feat_enc_i)
        self.dropout_input     = nn.Dropout(0.0)
        self.encoder           = TransformerEncoder(cfg)

        # Remnants (should be kept for backward compatibility)
        self.dropout_features = nn.Dropout(0.0)
        self.mask_emb = nn.Parameter(FloatTensor(feat_enc_i).uniform_())

    def forward(self):
        """Not used."""
        raise RuntimeError("Not implemented.")

    def extract_features(self, source: Tensor, output_layer: int | None = None, conv_only: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Extract WavLM feature series.
        
        Args:
            source        :: (B, T)           - The waveform
            output_layer                      - Target transformer layer number (index+1)
        Returns:
            features      :: (B, Frame, Feat) -
            x             :: (B, Frame, Feat) -
            layer_results :: 
        """

        # Conv-LN-Proj-Do :: (B, T) -> (B, Feat, Frame) -> (B, Frame, Feat) -> (B, Frame, Feat) -> (B, Frame, Feat)
        features = self.feature_extractor(source)
        features = self.layer_norm(features.transpose(1, 2))
        features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        if conv_only:
            return features, torch.empty(1), torch.empty(1)

        # Transformer :: (B, Frame, Feat) -> (B, Frame, Feat) - Feature from target layer & Features from intermediate layers
        x, layer_results = self.encoder(features, padding_mask=None, layer=None if output_layer is None else output_layer - 1)

        return features, x, layer_results


class ConvFeatureExtractionModel(nn.Module):
    """Convolutional feature extractor. Output match is confirmed."""

    def __init__(self):
        """Init."""
        super().__init__() # pyright: ignore [reportUnknownMemberType]

        # total stride 400@16kHz (25msec), total hop 320@16kHz (20msec), matched with the paper
        conv_layers: list[tuple[int, int, int, int]] = [
        #    c_i  c_o   k  s
            (  1, 512, 10, 5),
            (512, 512,  3, 2),
            (512, 512,  3, 2),
            (512, 512,  3, 2),
            (512, 512,  3, 2),
            (512, 512,  2, 2),
            (512, 512,  2, 2),
        ]

        def block(n_in: int, n_out: int, k: int, stride: int):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=0.0),
                nn.Sequential(
                    TransposeLast(),
                    Fp32LayerNorm(n_out, elementwise_affine=True),
                    TransposeLast(),
                ),
                nn.GELU(),
            )

        self.conv_layers = nn.ModuleList()
        for c_in, c_out, kernel, stride in conv_layers:
            self.conv_layers.append(block(c_in, c_out, kernel, stride))

    def forward(self, series: Tensor):
        """Forward a batch w/o padding :: (B, T) -> (B, Feat=1, T) -> (B, Feat, Frame)"""
        series = series.unsqueeze(1)
        for conv in self.conv_layers:
            series = conv(series)
        return series


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    has_relative_attention_bias=(self.relative_position_embedding and i == 0),
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                )
                for i in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer: int | None = None):
        """
        Args:
            layer - Target transformer layer index
        Returns:
            x             - Feature series from target layer
            layer_results - Feature series from intermediate layers
        """

        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer: int | None = None):
        """
        Args:
            tgt_layer - Target transformer layer index (None means final layer)
        Returns:
            x             - Feature series from target layer
            layer_results - Feature series from intermediate layers
        """

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # layer_results :: ()[]
        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        # Apply Layers with LayerDropout
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, self_attn_mask=streaming_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            # Stop at target layer
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
            has_relative_attention_bias: bool = False,
            num_buckets: int = 0,
            max_distance: int = 0,
            rescale_init: bool = False,
            gru_rel_pos: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: Tensor,
            self_attn_mask: Tensor = None,
            self_attn_padding_mask: Tensor = None,
            need_weights: bool = False,
            pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias

