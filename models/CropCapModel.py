from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

import opts
from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    """hybrid dependency and cross sight."""
    def __init__(self, multilevel_encoder, refine_encoder, decoder, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.multilevel_encoder = multilevel_encoder
        # self.crosssight_encoder = crosssight_encoder
        self.refine_encoder = refine_encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
    """hybrid dependency and cross sight."""

    def forward(self, grid_0, grid_1, grid_2, grid_3, tgt, tgt_mask):
        "Take in and process masked src and target sequences."

        """hybrid dependency and cross sight."""
        # x_0, x_1, x_2, x_3 = self.multilevel_encode(grid_0, grid_1, grid_2, grid_3)
        return self.decode(self.refine_encode(self.multilevel_encode(grid_0, grid_1, grid_2, grid_3)), tgt, tgt_mask)

    """hybrid dependency and cross sight."""
    def multilevel_encode(self, x_0, x_1, x_2, x_3):
        return self.multilevel_encoder(x_0, x_1, x_2, x_3)
    """hybrid dependency and cross sight."""

    """hybrid dependency and cross sight."""
    # def crosssight_encode(self, x_0, x_1, x_2, x_3):
    #     return self.crosssight_encoder(x_0, x_1, x_2, x_3)
    """hybrid dependency and cross sight."""

    def refine_encode(self, features, att_mask=None):
        return self.refine_encoder(features, att_mask)

    def decode(self, memory, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiLevelEncoder(nn.Module):
    """ building hybrid dependency. """
    def __init__(self, dropout):
        super(MultiLevelEncoder, self).__init__()
        self.adjust_linears_16 = nn.ModuleList([nn.Linear(i, (i // 16)) for i in [384, 768, 1536, 1536]])
        self.norms = nn.ModuleList([nn.LayerNorm(i) for i in [384, 192, 96, 96]])
        self.dropout = nn.Dropout(dropout)

        self.fuse_global = nn.Linear(144, 1, bias=False)
        self.fuse_se_0 = nn.Linear(768, 768 // 6, bias=False)
        self.fuse_se_1 = nn.Linear(768 // 6, 768, bias=False)
        self.norm_fuse = nn.LayerNorm(768)

        self.fuse_linear = nn.Linear(768, 512)

    def merging(self, x):
        B, HW, C = x.shape
        x = x.view(B, int(HW ** 0.5), int(HW ** 0.5), C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        return x

    def merges(self, x, i):
        for _ in range(i):
            x = self.merging(x)
        return x

    def down_sampling_to_same_size(self, x_0, x_1, x_2, x_3):
        x_0, x_1, x_2, x_3 = [self.dropout(l(x)) for l, x in zip(self.adjust_linears_16, (x_0, x_1, x_2, x_3))]

        x_0 = self.merges(x_0, 2)
        x_0 = self.norms[0](x_0)  # [40, 144, 384]

        x_1 = self.merges(x_1, 1)
        x_1 = self.norms[1](x_1)

        x_1 = torch.cat([x_1, x_0], -1)  # [40, 144, 192]

        x_2 = self.norms[2](x_2)

        x_2 = torch.cat([x_2, x_1], -1)  # [40, 144, 96]

        x_3 = self.norms[3](x_3)  # [40, 144, 96]
        return torch.cat([x_3, x_2], -1)

    def se_fuse(self, x):
        y = self.fuse_global(x.transpose(-1, -2))  # [10, 768, 1]
        y = self.fuse_se_0(y.transpose(-1, -2))  # [10, 1, 128]
        y = self.dropout(self.fuse_se_1(y))  # [10, 1, 768]
        return x + self.norm_fuse(x * y)

    def forward(self, x_0, x_1, x_2, x_3):
        return self.dropout(self.fuse_linear(self.se_fuse(self.down_sampling_to_same_size(x_0, x_1, x_2, x_3))))


class RefineEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer_segatt, N_segatt, layer_refine, N_refine):
        super(RefineEncoder, self).__init__()
        self.layers_segatt = clones(layer_segatt, N_segatt)
        self.norm_segatt = LayerNorm(layer_segatt.size)

        self.layers_refine = clones(layer_refine, N_refine)
        self.norm_refine = LayerNorm(layer_refine.size)

    def forward(self, x, mask):
        # print('199 mask is :', mask)
        "Pass the input (and mask) through each layer in turn."
        for layer_segatt in self.layers_segatt:
            x = self.norm_segatt(layer_segatt(x, mask))

        for layer_refine in self.layers_refine:
            x = layer_refine(x, mask)
        return self.norm_refine(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LayerNorm_C(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm_C, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        std = x.std(-2, keepdim=True)

        return (((self.a_2 * (x - mean).transpose(1, 2)).transpose(1, 2) / (std + self.eps)).transpose(1, 2) + self.b_2).transpose(1, 2)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class RefineSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(RefineSublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(x))


class RefineEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(RefineEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_0 = RefineSublayerConnection(size, dropout)
        self.sublayer_1 = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer_0(x, lambda x: self.self_attn(x, mask))
        return self.sublayer_1(x, self.feed_forward)


class SegAttEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, seg_attn, feed_forward, dropout):
        super(SegAttEncoderLayer, self).__init__()
        self.self_seg_attn = seg_attn
        self.feed_forward = feed_forward
        self.sublayer_0 = RefineSublayerConnection(size, dropout)
        self.sublayer_1 = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer_0(x, lambda x: self.self_seg_attn(x, mask))
        return self.sublayer_1(x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


def attentionc(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [10, 4, 512, 512]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h_p, h_c, d_model, s_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h_p == 0
        assert s_model % h_c == 0
        # assume d_v always equals d_k
        self.d_k = d_model // h_p
        self.s_k = s_model // h_c
        self.h_p = h_p
        self.h_c = h_c
        self.linears_p = clones(nn.Linear(d_model, d_model), 3)
        self.linears_c = clones(nn.Linear(s_model, s_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.norm_p = LayerNorm(d_model)
        self.norm_c = LayerNorm_C(s_model)

    def forward(self, x, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = x.size(0)
        x_p = self.norm_p(x)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_p, key_p, value_p = [l(x).view(nbatches, -1, self.h_p, self.d_k).transpose(1, 2)
                                   for l, x in zip(self.linears_p, (x_p, x_p, x_p))]

        # 2) Apply attention on all the projected vectors in batch.
        x_p = attention(query_p, key_p, value_p, mask=mask, dropout=self.dropout)  # [10, 8, 144, 64]
        x_p = x_p.transpose(1, 2).contiguous().view(nbatches, -1, self.h_p * self.d_k)
        x_p = self.norm_c(x_p)

        # 3) Do all the linear projections in batch from s_model => h x d_k
        query_c, key_c, value_c = [l(x.transpose(-1, -2)).view(nbatches, -1, self.h_c, self.s_k).transpose(1, 2)
                                   for l, x in zip(self.linears_c, (x_p, x_p, x_p))]

        x_c = attentionc(query_c, key_c, value_c, mask=mask, dropout=self.dropout)  # [10, 2, 512, 72]

        # 3) "Concat" using a view and apply a final linear.
        x_c = x_c.transpose(1, 2).contiguous().view(nbatches, -1, self.h_c * self.s_k)
        return self.linears_c[-1](x_c).transpose(-1, -2)


class SegAttMultiHeadedAttention(nn.Module):
    def __init__(self, seg_lenth_spatial, seg_lenth_channel, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(SegAttMultiHeadedAttention, self).__init__()
        self.d_model = d_model
        self.seg_length_spatial = seg_lenth_spatial
        self.seg_length_channel = seg_lenth_channel
        self.linears_spatial_seg = clones(nn.Linear(seg_lenth_spatial, seg_lenth_spatial), 3)
        self.linears_channel_seg = clones(nn.Linear(seg_lenth_channel, seg_lenth_channel), 3)
        self.linears = clones(nn.Linear(d_model, d_model), 1)

        self.norm_spatial_seg = LayerNorm(seg_lenth_spatial)
        self.norm_channel_seg = LayerNorm(seg_lenth_channel)

        self.dropout = nn.Dropout(p=dropout)

    def spatail_segatt(self, x, nbatches, mask=None):
        x = x.view(nbatches, self.seg_length_spatial, -1).transpose(-1, -2)
        x = self.norm_spatial_seg(x)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_s_seg, key_s_seg, value_s_seg = [l(x) for l, x in zip(self.linears_spatial_seg, (x, x, x))]

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query_s_seg, key_s_seg, value_s_seg, mask=mask, dropout=self.dropout)  # [10, 8, 144, 64]

        return x.transpose(-1, -2).contiguous().view(nbatches, -1, self.d_model)

    def channel_segatt(self, x, nbatches, mask=None):
        x = x.view(nbatches, -1, self.seg_length_channel)
        x = self.norm_channel_seg(x)

        # 3) Do all the linear projections in batch from s_model => h x d_k
        query_c_seg, key_c_seg, value_c_seg = [l(x) for l, x in
                                               zip(self.linears_channel_seg, (x, x, x))]

        x = attention(query_c_seg, key_c_seg, value_c_seg, mask=mask, dropout=self.dropout)  # [10, 2, 512, 72]

        # 3) "Concat" using a view and apply a final linear.
        return x.view(nbatches, -1, self.d_model)

    def forward(self, x, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = x.size(0)
        x = self.spatail_segatt(x, nbatches, mask)
        return self.dropout(self.linears[-1](self.channel_segatt(x, nbatches, mask)))
        # return self.dropout(self.linears[-1](self.channel_segatt(x, nbatches, mask) + self.spatail_segatt(x, nbatches, mask)))


class DeMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(DeMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class HFORMERModel(CaptionModel):
    def make_model(self, src_vocab, tgt_vocab, seg_lenth_spatial=36, seg_lenth_channel=64, N_segatt=1, N_refine=1,
                   DeN=6, d_model=512, s_model=144, d_ff=2048, h_p=8, h_c=4, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy

        attn = MultiHeadedAttention(h_p, h_c, d_model, s_model)
        segattn = SegAttMultiHeadedAttention(seg_lenth_spatial, seg_lenth_channel, d_model, dropout)
        deattn = DeMultiHeadedAttention(h_p, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        """hybrid dependency and cross sight."""
        model = EncoderDecoder(MultiLevelEncoder(dropout),
                               RefineEncoder(SegAttEncoderLayer(d_model, c(segattn), c(ff), dropout), N_segatt,
                                             RefineEncoderLayer(d_model, c(attn), c(ff), dropout), N_refine),
                               Decoder(DecoderLayer(d_model, c(deattn), c(deattn), c(ff), dropout), DeN),
                               nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                               Generator(d_model, tgt_vocab))
        """hybrid dependency and cross sight."""

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(HFORMERModel, self).__init__()
        self.opt = opt

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.att_feat_size = opt.att_feat_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0  # Schedule sampling probability

        tgt_vocab = self.vocab_size + 1  # add the start token into the vocabs
        self.model = self.make_model(0, tgt_vocab, N_refine=opt.num_layers, N_segatt=opt.num_layers,
                                     d_model=opt.input_encoding_size, d_ff=opt.rnn_size)

    def _prepare_input(self, x_0, x_1, x_2, x_3, seq=None, images_masks=None):
        """[40, 2304, 384], [40, 576, 768], [40, 144, 1536], [40, 144, 1536]"""
        x_0, x_1, x_2, x_3 = pack_wrapper(lambda x: x, x_0, x_1, x_2, x_3, images_masks)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] = 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        # return image, seq, seq_mask
        return x_0, x_1, x_2, x_3, seq, seq_mask

    def _forward(self, x_0, x_1, x_2, x_3, seq):
        """
        images: [B*5, 3, 384, 384]
        seq: [B*5, 17]
        seq_masks: [B*5, 17, 17] ==> bool type: subsequent_mask with np.triu()
        """
        # print("451 x_0, x_1, x_2, x_3 size are :{}, {}, {}, {}".format(x_0.size(), x_1.size(), x_2.size(), x_3.size()))
        """[40, 2304, 384], [40, 576, 768], [40, 144, 1536], [40, 144, 1536]"""
        x_0, x_1, x_2, x_3, seq, seq_masks = self._prepare_input(x_0, x_1, x_2, x_3, seq)
        """[40, 2304, 384], [40, 576, 768], [40, 144, 1536], [40, 144, 1536]"""
        # print("453 x_0, x_1, x_2, x_3 size are :{}, {}, {}, {}".format(x_0.size(), x_1.size(), x_2.size(), x_3.size()))

        out = self.model(x_0, x_1, x_2, x_3, seq, seq_masks)

        outputs = self.model.generator(out)
        return outputs

    def get_logprobs_state(self, it, memory, state):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, ys, subsequent_mask(ys.size(1)).to(memory.device))
        logprobs = self.model.generator(out[:, -1])

        return logprobs, [ys.unsqueeze(0)]

    def _sample_beam(self, x_0, x_1, x_2, x_3, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = x_0.size(0)

        x_0, x_1, x_2, x_3, seq, seq_masks = self._prepare_input(x_0, x_1, x_2, x_3)

        # memory = self.model.refine_encode(self.model.fuse_encode(x_0, x_1, x_2, x_3))
        # x_0, x_1, x_2, x_3 = self.model.hybriddependency_encode(x_0, x_1, x_2, x_3)
        memory = self.model.refine_encode(self.model.multilevel_encode(x_0, x_1, x_2, x_3))

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            tmp_memory = memory[k:k+1].expand(*((beam_size,) + memory.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = x_0.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_memory, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample_(self, x_0, x_1, x_2, x_3, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(x_0, x_1, x_2, x_3, opt)

        if sample_max:
            with torch.no_grad():
                seq_, seqLogprobs_ = self._sample_(x_0, x_1, x_2, x_3, opt)

        batch_size = x_0.shape[0]

        x_0, x_1, x_2, x_3, seq, seq_masks = self._prepare_input(x_0, x_1, x_2, x_3)

        # memory = self.model.refine_encode(self.model.fuse_encode(x_0, x_1, x_2, x_3))
        # x_0, x_1, x_2, x_3 = self.model.hybriddependency_encode(x_0, x_1, x_2, x_3)
        memory = self.model.refine_encode(self.model.multilevel_encode(x_0, x_1, x_2, x_3))

        ys = torch.zeros((batch_size, 1), dtype=torch.long).to(x_0.device)

        seq = x_0.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = x_0.new_zeros(batch_size, self.seq_length)

        for i in range(self.seq_length):
            out = self.model.decode(memory, ys, subsequent_mask(ys.size(1)).to(x_0.device))
            logprob = self.model.generator(out[:, -1])
            if sample_max:
                sampleLogprobs, next_word = torch.max(logprob, dim=1)
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprob.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprob.data, temperature))
                next_word = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprob.gather(1, next_word)  # gather the logprobs at sampled positions

            seq[:, i] = next_word
            seqLogprobs[:, i] = sampleLogprobs
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        assert (seq*((seq_ > 0).long()) == seq_).all(), 'seq doens\'t match'
        assert (seqLogprobs*((seq_ > 0).float()) - seqLogprobs_*((seq_ > 0).float())).abs().max() < 1e-5, 'logprobs doens\'t match'
        return seq, seqLogprobs

    def _sample(self, x_0, x_1, x_2, x_3, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(x_0, x_1, x_2, x_3, opt)

        batch_size = x_0.shape[0]

        x_0, x_1, x_2, x_3, seq, seq_masks = self._prepare_input(x_0, x_1, x_2, x_3)

        state = None
        # memory = self.model.refine_encode(self.model.fuse_encode(x_0, x_1, x_2, x_3))
        # x_0, x_1, x_2, x_3 = self.model.hybriddependency_encode(x_0, x_1, x_2, x_3)
        memory = self.model.refine_encode(self.model.multilevel_encode(x_0, x_1, x_2, x_3))

        seq = x_0.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = x_0.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = x_0.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, memory, state)
            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:, t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs
