# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.core.torch_agent import Beam
from parlai.core.dict import DictionaryAgent
import os
from parlai.agents.transformer import Constants
import numpy as np
from .layers import EncoderLayer, DecoderLayer
import ipdb


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    # def __init__(
    #         self,
    #         n_src_vocab, len_max_seq,
    #         d_word_vec=512, d_model=512, d_inner=2048,
    #         n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
    #         tgt_emb_prj_weight_sharing=True,
    #         emb_src_tgt_weight_sharing=True):
    def __init__(self, n_src_vocab, opt, longest_label=1):

        super().__init__()
        n_tgt_vocab = n_src_vocab
        len_max_seq= opt['max_token_seq_len']
        d_word_vec = opt['d_word_vec']
        d_model = opt['d_model']
        d_inner = opt['d_inner']
        n_layers = opt['n_layers']
        n_head = opt['n_head']
        d_k = opt['d_k']
        d_v = opt['d_v']
        dropout = opt['dropout']
        tgt_emb_prj_weight_sharing = opt['tgt_emb_prj_weight_sharing']
        emb_src_tgt_weight_sharing = opt['emb_src_tgt_weight_sharing']
        self.longest_label = longest_label

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, xs, ys, cands=None, cand_indices=None, prev_enc=None, rank_during_training=False, beam_size=1, topk=1):
        
        nbest_beam_preds, nbest_beam_scores = None, None

        bsize = xs.shape[0]
        src_seq = xs
        tgt_seq = ys

        # add position embedding
        src_pos = torch.zeros(src_seq.shape, dtype=torch.int64)
        for i in range(src_seq.shape[0]):
            seq = src_seq[i]
            pos = [pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(seq)]
            src_pos[i] = torch.tensor(pos)

        tgt_pos = torch.zeros(tgt_seq.shape, dtype=torch.int64)
        for i in range(tgt_seq.shape[0]):
            seq = tgt_seq[i]
            pos = [pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(seq)]
            tgt_pos[i] = torch.tensor(pos)
        ipdb.set_trace()
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        #output_len = seq_logit.shape[0] // bsize
        #scores = seq_logit.max(1)[0].view(bsize, output_len)
        #predictions = seq_logit.max(1)[1].view(bsize, output_len)
        scores = seq_logit.max(2)[0]
        predictions = seq_logit.max(2)[1]

        cand_preds, cand_scores = None, None
        # TODO candidate ranking


        # ret = (predictions, scores, cand_preds, cand_scores, encoder_states, nbest_beam_preds, nbest_beam_scores)
        ret = (predictions, scores, cand_preds)

        # return seq_logit.view(-1, seq_logit.size(2))
        return ret


class Ranker(object):
    def __init__(self, decoder, padding_idx=0, attn_type='none'):
        super().__init__()
        self.decoder = decoder
        self.NULL_IDX = padding_idx
        self.attn_type = attn_type

    def forward(self, cands, cand_inds, decode_params):
        start, hidden, enc_out, attn_mask = decode_params

        hid, cell = (hidden, None) if isinstance(hidden, torch.Tensor) else hidden
        if len(cand_inds) != hid.size(1):
            cand_indices = start.detach().new(cand_inds)
            hid = hid.index_select(1, cand_indices)
            if cell is None:
                hidden = hid
            else:
                cell = cell.index_select(1, cand_indices)
                hidden = (hid, cell)
            enc_out = enc_out.index_select(0, cand_indices)
            if attn_mask is not None:
                attn_mask = attn_mask.index_select(0, cand_indices)

        cand_scores = []

        for i in range(len(cands)):
            curr_cs = cands[i]

            n_cs = curr_cs.size(0)
            starts = start.expand(n_cs).unsqueeze(1)
            scores = 0
            seqlens = 0
            # select just the one hidden state
            if isinstance(hidden, torch.Tensor):
                nl = hidden.size(0)
                hsz = hidden.size(-1)
                cur_hid = hidden.select(1, i).unsqueeze(1).expand(nl, n_cs, hsz)
            else:
                nl = hidden[0].size(0)
                hsz = hidden[0].size(-1)
                cur_hid = (hidden[0].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous(),
                           hidden[1].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous())

            cur_enc, cur_mask = None, None
            if attn_mask is not None:
                cur_mask = attn_mask[i].unsqueeze(0).expand(n_cs, attn_mask.size(-1))
                cur_enc = enc_out[i].unsqueeze(0).expand(n_cs, enc_out.size(1), hsz)
            # this is pretty much copied from the training forward above
            if curr_cs.size(1) > 1:
                c_in = curr_cs.narrow(1, 0, curr_cs.size(1) - 1)
                xs = torch.cat([starts, c_in], 1)
            else:
                xs, c_in = starts, curr_cs
            if self.attn_type == 'none':
                preds, score, cur_hid = self.decoder(xs, cur_hid, cur_enc, cur_mask)
                true_score = F.log_softmax(score, dim=2).gather(
                    2, curr_cs.unsqueeze(2))
                nonzero = curr_cs.ne(0).float()
                scores = (true_score.squeeze(2) * nonzero).sum(1)
                seqlens = nonzero.sum(1)
            else:
                for i in range(curr_cs.size(1)):
                    xi = xs.select(1, i)
                    ci = curr_cs.select(1, i)
                    preds, score, cur_hid = self.decoder(xi, cur_hid, cur_enc, cur_mask)
                    true_score = F.log_softmax(score, dim=2).gather(
                        2, ci.unsqueeze(1).unsqueeze(2))
                    nonzero = ci.ne(0).float()
                    scores += true_score.squeeze(2).squeeze(1) * nonzero
                    seqlens += nonzero

            scores /= seqlens  # **len_penalty?
            cand_scores.append(scores)

        max_len = max(len(c) for c in cand_scores)
        cand_scores = torch.cat([pad(c, max_len).unsqueeze(0) for c in cand_scores], 0)
        preds = cand_scores.sort(1, True)[1]
        return preds, cand_scores


class Linear(nn.Module):
    """Custom Linear layer which allows for sharing weights (e.g. with an
    nn.Embedding layer).
    """
    def __init__(self, in_features, out_features, bias=True,
                 shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (shared_weight.size(0) != out_features or
                    shared_weight.size(1) != in_features):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        if self.shared:
            # detach weight to prevent gradients from changing weight
            # (but need to detach every time so weights are up to date)
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class RandomProjection(nn.Module):
    """Randomly project input to different dimensionality."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features),
                                requires_grad=False)  # fix weights
        self.reset_parameters()

    def reset_parameters(self):
        # experimentally: std=1 appears to affect scale too much, so using 0.1
        self.weight.data.normal_(std=0.1)
        # other init option: set randomly to 1 or -1
        # self.weight.data.bernoulli_(self.weight.fill_(0.5)).mul_(2).sub_(1)

    def forward(self, input):
        return F.linear(input, self.weight)


class AttentionLayer(nn.Module):
    def __init__(self, attn_type, hidden_size, emb_size, bidirectional=False,
                 attn_length=-1, attn_time='pre'):
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hidden_size
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = emb_size
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim,
                                          bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, enc_out, attn_mask=None):
        if self.attention == 'none':
            return xes

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        if self.attention == 'local':
            if enc_out.size(1) > self.max_length:
                offset = enc_out.size(1) - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)
            if attn_weights.size(1) > enc_out.size(1):
                attn_weights = attn_weights.narrow(1, 0, enc_out.size(1))
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                hid = hid.expand(last_hidden.size(0),
                                 enc_out.size(1),
                                 last_hidden.size(1))
                h_merged = torch.cat((enc_out, hid), 2)
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                if hid.size(2) != enc_out.size(2):
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            elif self.attention == 'general':
                hid = self.attn(hid)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            # calculate activation scores
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask -= (1 - attn_mask) * 1e20
            attn_weights = F.softmax(attn_w_premask, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        output = F.tanh(self.attn_combine(merged).unsqueeze(1))
        return output
