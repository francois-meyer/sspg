# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax, FairseqDropout
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence


DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5
LOGINF = 1000000.0


def map_chars2lex(tgt_dict, tgt_lex):
    chars2lex = {}
    for lex_id, subword in enumerate(tgt_lex.symbols):
        if lex_id < tgt_lex.nspecial:
            special_char_id = (tgt_dict.indices[subword], )
            chars2lex[special_char_id] = lex_id
        else:
            seg_chars = tuple(subword)
            seg_char_ids = tuple((tgt_dict.indices[char] for char in seg_chars))
            chars2lex[seg_char_ids] = lex_id
    return chars2lex


def map_prechars2lex(chars2lex):
    char_indices = list(chars2lex.keys())
    prechars2lex = {}
    for chars in char_indices:
        seg_len = len(chars)
        if seg_len == 1:
            continue
        for end in range(1, seg_len):
            prechars = chars[0: end]
            if prechars in prechars2lex:
                prechars2lex[prechars].append(chars2lex[chars])
            else:
                prechars2lex[prechars] = [chars2lex[chars]]
    return prechars2lex


def map_tokens2lex(src_dict, tgt_lex):
    """
    Map data tokens that are in the lexicon to lexicon indices.
    2 data tokens can map to the same lexicon index if they are only differentiated by starting a word or being in the
    middle e.g. @@he and he will both map to the lexicon subword he.
    """
    tokens2lex = {}
    for token_id, token in enumerate(src_dict.symbols):
        if token_id < src_dict.nspecial:
            continue
        if token.startswith("@@"):
            token = token[2:]
        if token in tgt_lex.symbols:
            tokens2lex[token_id] = tgt_lex.indices[token]
    return tokens2lex


def map_tokens2chars(src_dict, tgt_dict):
    """
    Map source token vocabulary to character sequence in the target character vocabulary.
    """
    tokens2chars = {}
    for token_id, token in enumerate(src_dict.symbols):
        if token_id < src_dict.nspecial:
            continue
        if token.startswith("@@"):
            token = token[2:]
        if all((char in tgt_dict.indices for char in token)):
            tokens2chars[token_id] = tuple((tgt_dict.indices[char] for char in token))
    return tokens2chars


def map_prechars2tokens(chars2tokens):
    char_indices = list(chars2tokens.keys())
    prechars2tokens = {}
    for chars in char_indices:
        seg_len = len(chars)
        if seg_len == 1:
            continue
        for end in range(1, seg_len):
            prechars = chars[0: end]
            if prechars in prechars2tokens:
                prechars2tokens[prechars].append(chars2tokens[chars])
            else:
                prechars2tokens[prechars] = [chars2tokens[chars]]
    return prechars2tokens


def map_pretokens2lex(tokens2chars, prechars2lex):
    pretokens2lex = {}
    for token in tokens2chars:
        chars = tokens2chars[token]
        if chars in prechars2lex:
            pretokens2lex[token] = prechars2lex[chars]
    return pretokens2lex


@register_model("ssd2t")
class SubwordSegmentalData2Text(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Custom arguments
        parser.add_argument('--decoder-copy', action='store_true',
                            help='add conditional copy mechanism to decoder')
        parser.add_argument('--source-position-markers', type=int, metavar='N',
                            help='dictionary includes N additional items that '
                                 'represent an OOV token at a particular input '
                                 'position')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError("--encoder-layers must match --decoder-layers")

        max_source_positions = getattr(
            args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )
        max_target_positions = getattr(
            args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )
        if getattr(args, "source_position_markers", None) is None:
            args.source_position_markers = max_source_positions

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError("--share-all-embeddings requires a joint dictionary")
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embed not compatible with---decoder-embed-path"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to "
                    "match --decoder-embed-dim"
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim,
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
            args.decoder_embed_dim != args.decoder_out_embed_dim
        ):
            raise ValueError(
                "--share-decoder-input-output-embeddings requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
        )
        decoder = LSTMDecoder(
            tgt_dictionary=task.target_dictionary,
            tgt_lexicon=task.target_lexicon,
            src_dictionary=task.source_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=utils.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            residuals=False,
            # Custom arguments
            copy=args.decoder_copy,
            source_position_markers=args.source_position_markers,
        )
        return cls(encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        mode: Optional[str] = "forward",
    ):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            mode=mode,
        )
        return decoder_out

    def generate_mode(self):
        self.decoder.generate = True


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        left_pad=True,
        pretrained_embed=None,
        padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in_module = FairseqDropout(
            dropout_in * 1.0, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out * 1.0, module_name=self.__class__.__name__
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx, dictionary.unk())
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        enforce_sorted: bool = True,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.cpu(), enforce_sorted=enforce_sorted
        )

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_idx * 1.0
        )
        x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return tuple(
            (
                x,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                final_cells,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
                src_tokens,
            )
        )

    def combine_bidir(self, outs, bsz: int):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def reorder_encoder_out(
        self, encoder_out: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], new_order
    ):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
                encoder_out[3].index_select(1, new_order),
                encoder_out[4].index_select(0, new_order),
            )
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self,
        tgt_dictionary,
        tgt_lexicon,
        src_dictionary,
        embed_dim=512,
        hidden_size=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention=True,
        encoder_output_units=512,
        pretrained_embed=None,
        share_input_output_embed=False,
        adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        residuals=False,
        copy=False,
        source_position_markers=0,
        max_seg_len=5,
    ):
        super().__init__(tgt_dictionary)
        self.dropout_in_module = FairseqDropout(
            dropout_in * 1.0, module_name=self.__class__.__name__
        )
        self.dropout_out_module = FairseqDropout(
            dropout_out * 1.0, module_name=self.__class__.__name__
        )
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers

        self.adaptive_softmax = None
        self.num_char_embeddings = len(tgt_dictionary)

        self.padding_idx = tgt_dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(self.num_char_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    input_size=input_feed_size + embed_dim
                    if layer == 0
                    else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )

        if attention:
            self.attention = AttentionLayer(
                hidden_size, encoder_output_units, hidden_size, bias=False
            )
        else:
            self.attention = None

        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, hidden_size)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                self.num_embeddings,
                hidden_size,
                adaptive_softmax_cutoff,
                dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, self.num_char_embeddings, dropout=dropout_out)

        # Subword segmental components
        self.max_seg_len = max_seg_len
        self.char_decoder = LSTMCharDecoder(
            vocab_size=len(tgt_dictionary.symbols),
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_out
        )
        eom_word = "<eom>"
        self.eom_id = tgt_dictionary.indices[eom_word]
        self.reg_exp = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tgt_dictionary = tgt_dictionary
        self.src_dictionary = src_dictionary
        self.tgt_lexicon = tgt_lexicon

        self.char_nspecial = tgt_dictionary.nspecial
        self.char_nalpha = len([char for char in tgt_dictionary.symbols if char.isalpha()])
        self.chars2lex = map_chars2lex(tgt_dict=tgt_dictionary, tgt_lex=tgt_lexicon)
        self.lex2chars = {chars_ids: lex_id for lex_id, chars_ids in self.chars2lex.items()}
        self.prechars2lex = map_prechars2lex(self.chars2lex)
        self.lex_vocab_size = len(tgt_lexicon.symbols)
        self.src_vocab_size = len(src_dictionary.symbols)
        self.lex_decoder = LexDecoder(
            vocab_size=self.lex_vocab_size,
            hidden_size=embed_dim,
            num_layers=1,
            dropout=dropout_out
        )
        self.mixture_gate_func = nn.Linear(embed_dim, 1)
        self.generate = False
        self.decoding = None

        self.copy = copy
        if copy:
            self.num_types = len(tgt_lexicon)
            self.num_oov_types = source_position_markers

            self.gen_coef_mlp = nn.Linear(hidden_size, 1)
            self.num_oov_types = source_position_markers
            self.tokens2lex = map_tokens2lex(src_dict=src_dictionary, tgt_lex=tgt_lexicon)

            self.tokens2chars = map_tokens2chars(src_dict=src_dictionary, tgt_dict=tgt_dictionary)
            self.chars2tokens = {chars_ids: token_id for token_id, chars_ids in self.tokens2chars.items()}

            self.prechars2tokens = map_prechars2tokens(self.chars2tokens)
            self.pretokens2lex = map_pretokens2lex(self.tokens2chars, self.prechars2lex)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Tensor] = None,
        mode="forward"
    ):

        if not self.generate:
            return self.forward_train(prev_output_tokens, encoder_out, incremental_state, src_lengths, mode)
        else:
            return self.dynamic_decode(prev_output_tokens, encoder_out, incremental_state, src_lengths, mode)


    def dynamic_decode(self, prev_output_tokens, encoder_out, incremental_state, src_lengths, mode="forward"):
        if len(prev_output_tokens) == 2:
            beam_ids, prev_seg_ends = prev_output_tokens
            prev_history_encodings = None
            prev_attn_scores = None
        else:
            beam_ids, prev_seg_ends, prev_history_encodings, prev_attn_scores = prev_output_tokens
        beam_ids = beam_ids.to(self.device)

        if self.decoding == "separate":
            incremental_state = None

        # Encode character-level histories
        x, attn_scores, h = self.extract_features(
            beam_ids, encoder_out, incremental_state
        )  # x is the output features (with attention applied)

        history_encodings = x  # (batch_size, tgt_cur_len, embed_dim)
        if prev_history_encodings is not None:
            history_encodings = torch.cat([prev_history_encodings, history_encodings], dim=1)

        if prev_attn_scores is not None:
            attn_scores = torch.cat([prev_attn_scores, attn_scores], dim=1)

        src_token_ids = encoder_out[4]  # (batch_size * beam_size, src_len)

        if self.decoding == "separate":
            seg_lprobs, src_lprobs, top_char_segs, top_char_lprobs = self.next_seg_inference(
                beam_ids,
                history_encodings,
                prev_seg_ends,
                src_token_ids,
                attn_scores
            )
            return seg_lprobs, src_lprobs, top_char_segs, top_char_lprobs, history_encodings, attn_scores
        else:
            char_vocab_lprobs, char_eom_lprobs = self.inference(
                beam_ids,
                history_encodings,
                prev_seg_ends,
                src_token_ids,
                attn_scores
            )
            return char_vocab_lprobs, char_eom_lprobs, history_encodings, attn_scores

    def next_seg_inference(self,
        beam_ids,
        history_encodings,
        prev_seg_ends,
        src_token_ids,
        attn_scores
    ):
        num_beams = beam_ids.shape[0]
        src_len = src_token_ids.shape[1]
        char_vocab_size = self.char_decoder.vocab_size
        prev_seg_ends = torch.tensor(prev_seg_ends).to(self.device)
        log_attn_scores = torch.log(attn_scores + 1e-10)

        # Gather relevant history encoding for each sequence
        history_indices = prev_seg_ends.unsqueeze(-1)
        history_indices = history_indices.repeat(1, history_encodings.shape[-1])
        history_indices = history_indices.unsqueeze(1)
        history_encoding = torch.gather(history_encodings, dim=1, index=history_indices)
        history_encoding = history_encoding.squeeze(1)  # (batch_beam_size, embed_dim)
        log_gen_proportions, log_copy_proportions = self.compute_pointer_lcoefs(
            history_encoding
        )

        log_char_proportions, log_lex_proportions = self.compute_mix_lcoefs(
            history_encoding
        )

        attn_indices = prev_seg_ends.unsqueeze(-1)
        attn_indices = attn_indices.repeat(1, attn_scores.shape[-1])
        attn_indices = attn_indices.unsqueeze(1)
        log_attn_score = torch.gather(log_attn_scores, dim=1, index=attn_indices)

        # Compute lexicon lprobs
        lex_logits = self.lex_decoder(history_encoding)  # (batch_size * beam_size, lex_vocab_size)
        full_lex_logp = self.get_normalized_probs(net_output=(lex_logits,),
                                                  log_probs=True)  # in models.fairseq_decoder  # (lex_vocab_size)
        full_lex_logp = torch.cat((full_lex_logp, torch.full((full_lex_logp.shape[0], 1), fill_value=-LOGINF,
                                                             device=self.device)), dim=-1)

        # Extract copy probabilities for out-of-lexicon items
        full_src_logp = torch.full((num_beams, self.src_vocab_size),
                                        fill_value=-LOGINF, device=self.device)
        full_src_logp = torch.cat((full_src_logp, torch.full((full_src_logp.shape[0], 1),
                                                                       fill_value=-LOGINF, device=self.device)), dim=-1)

        for batch_num in range(num_beams):
            for src_pos in range(src_len):
                if src_token_ids[batch_num][src_pos] >= self.src_dictionary.nspecial:
                    full_src_logp[batch_num, src_token_ids[batch_num][src_pos]] = \
                        torch.logaddexp(full_src_logp[batch_num, src_token_ids[batch_num][src_pos]].clone(),
                                        log_attn_score[batch_num, -1, src_pos])

        # Greedy character decoding
        top_char_ids = []
        top_char_lprobs = []

        # Extract input embeddings for current segments
        # embedding: (seg_len + 1, num_segs, embedding_dim)
        # init_hidden_states: (1, num_segs, embedding_dim)
        seg_input_ids = torch.gather(beam_ids, dim=-1, index=prev_seg_ends.unsqueeze(-1))#.transpose(0, 1)
        seg_input_ids = torch.repeat_interleave(seg_input_ids, char_vocab_size, 0)

        # Extract history encoding up to last segment, prepare for all possible segments
        history_encoding = history_encoding.unsqueeze(0)  # (1, num_beams, embed_dim)
        history_encoding = torch.repeat_interleave(history_encoding, char_vocab_size, dim=1)  # (1, num_beams * char_vocab_size, embed_dim)

        summed_con_lprobs = []
        beam_indices = torch.tensor(list(range(0, num_beams * char_vocab_size, char_vocab_size))).to(self.device)
        char_vocab_ids = torch.arange(char_vocab_size).to(self.device).unsqueeze(1).repeat(num_beams, 1)
        seg_len = 1
        while seg_len <= self.max_seg_len:
            # Prep input embeddings for next char prediction
            seg_input_ids = torch.cat([seg_input_ids, char_vocab_ids], dim=1).transpose(0, 1)
            seg_input_embeddings = self.embed_tokens(seg_input_ids)

            # Output next char lprobs
            char_logits, _ = self.char_decoder(seg_input_embeddings, history_encoding)
            char_lprobs = self.get_normalized_probs(net_output=(char_logits,),
                                                    log_probs=True)  # (cur_seg_len+1, char_vocab_size, char_vocab_size)

            # Collect top continued chars
            con_lprobs = torch.index_select(char_lprobs[seg_len - 1], dim=0, index=beam_indices)
            con_lprobs = self.fix_lprobs(con_lprobs, cur_seg_len=seg_len)
            max_con_ids = torch.argmax(con_lprobs, dim=-1)
            max_con_lprobs = torch.max(con_lprobs, dim=-1)[0].unsqueeze(0)

            if seg_len > 1:
                max_summed_con_lprobs = torch.sum(
                    torch.stack([summed_con_lprobs[seg_len - 2], max_con_lprobs], dim=0), dim=0)
            else:
                max_summed_con_lprobs = max_con_lprobs
            summed_con_lprobs.append(max_summed_con_lprobs)

            # Collect top ended chars
            eom_lprobs = char_lprobs[seg_len, :, self.eom_id]

            con_lprobs = con_lprobs + summed_con_lprobs[seg_len - 1].transpose(0, 1)
            eom_lprobs = con_lprobs + eom_lprobs.view(num_beams, -1)
            max_eom_ids = torch.argmax(eom_lprobs, dim=-1)

            prev_con_ids = torch.index_select(seg_input_ids[1: seg_len], dim=1, index=beam_indices)
            max_eom_ids = torch.cat([prev_con_ids, max_eom_ids.unsqueeze(0)], dim=0)
            top_char_ids.append(max_eom_ids)

            max_eom_lprobs = torch.max(eom_lprobs, dim=-1)[0]
            top_char_lprobs.append(max_eom_lprobs)

            seg_input_ids = torch.index_select(seg_input_ids[0: -1], dim=1, index=beam_indices)
            seg_input_ids = torch.cat([seg_input_ids, max_con_ids.unsqueeze(0)], dim=0).transpose(0, 1)
            seg_input_ids = torch.repeat_interleave(seg_input_ids, char_vocab_size, 0)

            seg_len += 1

        seg_len = 1
        normalized_top_char_lprobs = []
        for seg_top_lprobs in top_char_lprobs:
            normalized_seg_lprobs = seg_top_lprobs / (seg_len + 1)
            normalized_top_char_lprobs.append(normalized_seg_lprobs)
            seg_len += 1
        top_char_lprobs = torch.stack(normalized_top_char_lprobs, dim=0)

        next_char_ids = []
        for beam_num in range(num_beams):
            next_char_ids.append([])
            for seg_len in range(self.max_seg_len):
                next_char_ids[-1].append(top_char_ids[seg_len][:, beam_num].tolist())

        top_char_lprobs = top_char_lprobs.transpose(0, 1)

        full_lex_logp = log_gen_proportions.unsqueeze(-1) + full_lex_logp
        full_src_logp = log_copy_proportions.unsqueeze(-1) + full_src_logp

        top_char_lprobs = log_char_proportions.unsqueeze(-1) + top_char_lprobs
        full_lex_logp = log_lex_proportions.unsqueeze(-1) + full_lex_logp
        full_src_logp = log_lex_proportions.unsqueeze(-1) + full_src_logp

        return full_lex_logp, full_src_logp, next_char_ids, top_char_lprobs

    def forward_train(self, prev_output_tokens, encoder_out, incremental_state, src_lengths, mode="forward"):
        x, attn_scores, h = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )  # x is the output features (with attention applied)

        # Prepare ids and history embeddings from Transformer encoder
        input_ids = torch.transpose(prev_output_tokens, 0, 1)  # [seq_len + 1, batch_size]
        target_ids = input_ids[1:, :]  # [seq_len, batch_size]
        input_embeddings = self.embed_tokens(input_ids)
        history_encodings = torch.transpose(x, 0, 1)[0: -1]  # [seq_len, batch_size, embed_dim]
        batch_size = target_ids.shape[1]
        seq_len = target_ids.shape[0]
        src_token_ids = encoder_out[4]

        # Collect segment input embeddings and target ids
        seg_embeddings, seg_target_ids, seg_lens, seq_lens = self.collect_segs(
            target_ids,
            input_embeddings,
            batch_size,
            seq_len
        )

        # Compute character-by-character probabilities
        char_logp, full_char_logp = self.compute_char_lprobs(
            history_encodings,
            seg_embeddings,
            seg_target_ids
        )

        # Compute lexicon probabilities
        if self.lex_vocab_size > 0:
            lex_logp = self.compute_lex_lprobs(
                history_encodings,
                target_ids,
                seq_len,
                src_token_ids,
                attn_scores
            )

            # Compute mixture coefficent
            log_char_proportions, log_lex_proportions = self.compute_mix_lcoefs(
                history_encodings
            )

            # Compute segment probabilities
            seg_logp = self.compute_seg_lprobs(
                char_logp,
                full_char_logp,
                seg_lens,
                batch_size,
                seq_len,
                lex_logp,
                log_char_proportions,
                log_lex_proportions
            )

        else:
            # Compute segment probabilities
            seg_logp = self.compute_seg_lprobs(
                char_logp,
                full_char_logp,
                seg_lens,
                batch_size,
                seq_len
            )

        # Dynamic programming
        if mode == "forward":
            # Compute marginals
            log_alpha = self.forward_pass(
                seg_logp,
                batch_size,
                seq_len
            )
            return log_alpha, -LOGINF, seq_lens
        elif mode == "segment":
            # Compute maximising segmentation
            log_gen_proportions, log_copy_proportions = self.compute_pointer_lcoefs(
                history_encodings
            )

            split_indices = self.viterbi(
                seg_logp,
                seq_lens
            )
            return split_indices

    def fix_lprobs(self, lprobs, cur_seg_len):

        # Next character cannot be <pad>
        lprobs[:, self.padding_idx] = -LOGINF

        # Next character cannot be eom itself
        lprobs[:, self.eom_id] = -LOGINF

        # Only alphabetic characters can be part of multi-character segment
        if cur_seg_len > 1:
            lprobs[:, 0: self.char_nspecial] = -LOGINF
            lprobs[:, self.char_nspecial + self.char_nalpha:] = -LOGINF

        # Segment cannot continue further if next character alread makes it as long as allowed
        if cur_seg_len > self.max_seg_len:
            lprobs[:] = -LOGINF

        return lprobs


    def compute_mix_lcoefs(self,
        history_encodings
    ):
        logits = self.mixture_gate_func(history_encodings).squeeze(-1)

        log_char_proportions = F.logsigmoid(logits)
        log_lex_proportions = F.logsigmoid(-logits)

        return log_char_proportions, log_lex_proportions

    def compute_pointer_lcoefs(self,
        history_encodings
    ):
        logits = self.gen_coef_mlp(history_encodings).squeeze(-1)

        log_gen_proportions = F.logsigmoid(logits)
        log_copy_proportions = F.logsigmoid(-logits)

        return log_gen_proportions, log_copy_proportions

    def collect_segs(
        self,
        target_ids,
        input_embeddings,
        batch_size,
        seq_len
    ):

        alpha_mask = (target_ids >= self.char_nspecial) & (target_ids < self.char_nspecial + self.char_nalpha)
        target_alphabet = torch.where(alpha_mask, True, False)
        pad_mask = torch.where(target_ids != self.padding_idx, True, False)
        seq_lens = torch.sum(pad_mask, dim=0)
        seg_embeddings = []
        seg_target_ids = []
        seg_lens = []

        for seg_end in range(self.max_seg_len - 1,
                             seq_len + self.max_seg_len - 1):  # end of first possible seg to end of sequence

            seg_target_alphas = target_alphabet[seg_end - self.max_seg_len + 1: seg_end + 1]
            seg_lens.append([])
            for seq_num in range(batch_size):
                if not seg_target_alphas[0][seq_num]:
                    seg_len = 1
                else:
                    for j in range(len(seg_target_alphas)):
                        if seg_target_alphas[j][seq_num]:
                            seg_len = j + 1
                        else:
                            break
                seg_lens[-1].append(seg_len)

            seg_embeddings.append(
                input_embeddings[   seg_end - self.max_seg_len + 1: seg_end + 2].clone())
            seg_target_ids.append(target_ids[seg_end - self.max_seg_len + 1: seg_end + 1].clone())

            if seg_end >= seq_len:  # pad with zero embeddings and pad ids
                seg_embeddings[-1] = torch.cat((seg_embeddings[-1], torch.zeros((seg_end - seq_len + 1, batch_size,
                                                                                 input_embeddings.shape[-1]),
                                                                                 device=self.device)))
                                                                                # device=self.device)))
                seg_target_ids[-1] = torch.cat((seg_target_ids[-1], torch.full((seg_end - seq_len + 1, batch_size),
                                                                               fill_value=self.padding_idx,
                                                                               device=self.device)))
                                                                               # device=self.device)))

        seg_embeddings = torch.stack(seg_embeddings, dim=1).view(self.max_seg_len + 1, -1, input_embeddings.shape[2])
        return seg_embeddings, seg_target_ids, seg_lens, seq_lens

    def compute_lex_lprobs(
        self,
        history_encodings,  # (seq_len, batch_size, embed_dim)
        target_ids,  # (seq_len, batch_size)
        seq_len,
        src_token_ids,  # (batch_size, src_len)
        attn_scores  # (batch_size, seq_len+1, src_len)
    ):
        batch_size = src_token_ids.shape[0]
        src_len = src_token_ids.shape[1]
        attn_scores = attn_scores[:, :-1, :]  # (batch_size, seq_len, src_len)
        log_attn_scores = torch.log(attn_scores + 1e-10)

        # Lexicon generation
        gen_logits = self.lex_decoder(history_encodings)
        full_gen_logp = self.get_normalized_probs(net_output=(gen_logits,),
                                                   log_probs=True)  # in models.fairseq_decoder
        full_gen_logp = torch.cat((full_gen_logp, torch.full((full_gen_logp.shape[0], full_gen_logp.shape[1], 1),
                                                             fill_value=-LOGINF, device=self.device)), dim=-1)

        if self.copy:
            # Map src tokens to lexicon ids
            src_lex_ids = torch.full(src_token_ids.shape, fill_value=-1, dtype=torch.long, device=self.device)
            for batch_num in range(batch_size):
                for src_pos in range(src_len):
                    if src_token_ids[batch_num][src_pos].item() in self.tokens2lex:
                        src_lex_ids[batch_num][src_pos] = self.tokens2lex[src_token_ids[batch_num][src_pos].item()]

            # Extract copy probabilities for lexicon items
            full_copy_lex_logp = torch.full_like(full_gen_logp, fill_value=-LOGINF, device=self.device)  # (seq_len, batch_size, lex_vocab_size)
            for batch_num in range(batch_size):
                for src_pos in range(src_len):
                    if src_lex_ids[batch_num][src_pos] != -1:
                        full_copy_lex_logp[:, batch_num, src_lex_ids[batch_num][src_pos]] = \
                            torch.logaddexp(full_copy_lex_logp[:, batch_num, src_lex_ids[batch_num][src_pos]].clone(),
                                            log_attn_scores[batch_num, :, src_pos])

            # Extract copy probabilities for out-of-lexicon items
            full_copy_unk_logp = torch.full((seq_len, batch_size, self.src_vocab_size),
                                            fill_value=-LOGINF, device=self.device)
            full_copy_unk_logp = torch.cat((full_copy_unk_logp, torch.full((full_copy_unk_logp.shape[0], full_copy_unk_logp.shape[1], 1),
                                                                 fill_value=-LOGINF, device=self.device)), dim=-1)

            for batch_num in range(batch_size):
                for src_pos in range(src_len):
                    if src_lex_ids[batch_num][src_pos] == -1 and src_token_ids[batch_num][src_pos] >= self.src_dictionary.nspecial:  # Check if unk
                        full_copy_unk_logp[:, batch_num, src_token_ids[batch_num][src_pos]] = \
                            torch.logaddexp(full_copy_unk_logp[:, batch_num, src_token_ids[batch_num][src_pos]].clone(),
                                            log_attn_scores[batch_num, :, src_pos])

            # Combine copy and gen coefficients
            log_gen_proportions, log_copy_proportions = self.compute_pointer_lcoefs(
                history_encodings
            )

        lex_logp ={}
        gen_logp = {}
        copy_lex_logp = {}
        copy_unk_logp = {}
        for seg_len in range(1, self.max_seg_len + 1):
            lex_logp[seg_len] = []
            gen_logp[seg_len] = []
            copy_lex_logp[seg_len] = []
            copy_unk_logp[seg_len] = []

            for seg_start in range(seq_len - (seg_len - 1)):
                # Extract char ids for segment
                seg_char_ids = target_ids[seg_start: seg_start + seg_len].T.tolist()
                seg_char_ids = [tuple(char_ids) for char_ids in seg_char_ids]

                # Extract lex ids based on chars
                seg_lex_ids = torch.LongTensor([self.chars2lex[char_ids] if char_ids in self.chars2lex
                                                else self.lex_vocab_size  # not in segment lexicon
                                                for char_ids in seg_char_ids]).to(self.device)

                # Extract lexicon generation lprobs
                gen_logp[seg_len].append(torch.gather(full_gen_logp[seg_start], dim=-1,
                                                      index=seg_lex_ids.unsqueeze(-1)).squeeze(-1))

                if self.copy:
                    # Extract lexicon copy lprobs
                    copy_lex_logp[seg_len].append(torch.gather(full_copy_lex_logp[seg_start], dim=-1,
                                                               index=seg_lex_ids.unsqueeze(-1)).squeeze(-1))

                    # Get source ids for any segments not in lexicon
                    seg_src_ids = torch.LongTensor([self.chars2tokens[char_ids] if char_ids in self.chars2tokens
                                                    and char_ids not in self.chars2lex
                                                    else self.src_vocab_size  # not in segment lexicon
                                                    for char_ids in seg_char_ids]).to(self.device)

                    copy_unk_logp[seg_len].append(torch.gather(full_copy_unk_logp[seg_start], dim=-1,
                                                              index=seg_src_ids.unsqueeze(-1)).squeeze(-1))

            lex_logp[seg_len] = torch.stack(gen_logp[seg_len], dim=0)
            if self.copy:
                copy_lex_logp[seg_len] = torch.stack(copy_lex_logp[seg_len], dim=0)
                copy_unk_logp[seg_len] = torch.stack(copy_unk_logp[seg_len], dim=0)
                # Combine lex and unk logps
                seg_copy_logp = torch.logsumexp(torch.stack([copy_lex_logp[seg_len], copy_unk_logp[seg_len]]), dim=0)

                seg_log_gen_proportions = log_gen_proportions[0: seq_len - (seg_len - 1)]
                seg_log_copy_proportions = log_copy_proportions[0: seq_len - (seg_len - 1)]

                gen_element = seg_log_gen_proportions + lex_logp[seg_len]
                copy_element = seg_log_copy_proportions + seg_copy_logp
                lex_logp[seg_len] = torch.logsumexp(torch.stack([gen_element, copy_element]), dim=0)


        return lex_logp

    def compute_char_lprobs(
        self,
        history_encodings,
        seg_embeddings,
        seg_target_ids
    ):
        seg_hidden_states = history_encodings.contiguous().view(1, -1, history_encodings.shape[2])
        char_logits, _ = self.char_decoder(seg_embeddings, seg_hidden_states)
        full_char_logp = self.get_normalized_probs(net_output=(char_logits,),
                                                   log_probs=True)  # in models.fairseq_decoder
        target_prob_ids = torch.stack(seg_target_ids, dim=1).view(self.max_seg_len, -1)
        char_logp = torch.gather(
            full_char_logp[0: self.max_seg_len],
            dim=-1,
            index=target_prob_ids.unsqueeze(-1)
        ).squeeze(-1)

        return char_logp, full_char_logp

    def compute_seg_lprobs(
        self,
        char_logp,
        full_char_logp,
        seg_lens,
        batch_size,
        seq_len,
        lex_logp=None,
        log_char_proportions=None,
        log_lex_proportions=None,
    ):
        seg_logp = {}
        for seg_len in range(1, self.max_seg_len + 1):
            end_batch_index = (seq_len - (seg_len - 1)) * batch_size

            seg_logp[seg_len] = torch.sum(char_logp[0: seg_len, 0: end_batch_index], dim=0) \
                                + full_char_logp[seg_len, 0: end_batch_index, self.eom_id]
            seg_logp[seg_len] = seg_logp[seg_len].view(-1, batch_size)

            # Keep only valid subwords
            valid_segs = torch.tensor(seg_lens[0: seq_len - (seg_len - 1)], device=self.device) >= seg_len
            seg_logp[seg_len] = torch.where(valid_segs, seg_logp[seg_len], torch.full_like(seg_logp[seg_len],
                                                                                               fill_value=-LOGINF))

            if self.lex_vocab_size > 0:
                # Calculate weighted average of character and lexical generation probabilities
                seg_log_char_proportions = log_char_proportions[0: seq_len - (seg_len - 1)]
                seg_log_lex_proportions = log_lex_proportions[0: seq_len - (seg_len - 1)]

                neginf_log_proportions = torch.full_like(seg_log_lex_proportions, fill_value=-LOGINF,
                                                         device=self.device)
                seg_log_lex_proportions = torch.where(lex_logp[seg_len] > -LOGINF, seg_log_lex_proportions,
                                                      neginf_log_proportions)

                char_element = seg_log_char_proportions + seg_logp[seg_len]
                lex_element = seg_log_lex_proportions + lex_logp[seg_len]
                seg_logp[seg_len] = torch.logsumexp(torch.stack([char_element, lex_element]), dim=0)

        return seg_logp

    def forward_pass(self,
        seg_logp,
        batch_size,
        seq_len
    ):
        # Compute alpha values and expected length factor
        log_alpha = torch.zeros((seq_len + 1, batch_size), device=self.device)
        for t in range(1, seq_len + 1):  # from alpha_1 to alpha_bptt_len
            range_j = list(range(max(0, t - self.max_seg_len), t))
            log_alphas_t = log_alpha[range_j[0]: range_j[-1] + 1]
            seg_logp_elements = []
            regs_t = torch.zeros((len(range_j), 1)).to(self.device)

            for j in range_j:
                seg_logp_elements.append(seg_logp[t - j][j])
                regs_t[j - range_j[0]] = torch.log(torch.FloatTensor([(t - j) ** self.reg_exp]))

            seg_logp_t = torch.stack(seg_logp_elements)
            log_alpha[t] = torch.logsumexp(log_alphas_t + seg_logp_t, dim=0)

        return log_alpha

    def viterbi(self,
        seg_logp,
        seq_lens
    ):
        split_indices = []
        for seq_num, seq_len in enumerate(seq_lens):
            # Compute alpha values and store backpointers
            bps = torch.zeros((self.max_seg_len, seq_len + 1), dtype=torch.long, device=self.device)
            max_logps = torch.full((self.max_seg_len, seq_len + 1), fill_value=0.0, device=self.device)
            for t in range(1, seq_len + 1):  # from alpha_1 to alpha_bptt_len
                alpha_sum_elements = []
                # print("t=", t)
                for j in range(max(0, t - self.max_seg_len), t):
                    # The current segment starts at j and ends at t
                    # The backpointer will point to the segment ending at j-1
                    max_bp = max(1, j)  # Maximum possible length of segment ending at j-1

                    # Compute the probability of the most likely sequence ending with the segment j-t (length t-j)
                    # For this most likely sequence ending at j-1, what is the final segment length?
                    bps[t - j - 1, t] = torch.argmax(max_logps[0: max_bp, j])

                    # What is the probability of the most likely sequence ending at t with last segment j-t?
                    max_logps[t - j - 1, t] = torch.max(max_logps[0: max_bp, j]) + seg_logp[t - j][j, seq_num]

            # Backtrack from final state of most likely path
            best_path = []
            k = torch.tensor(seq_len)
            bp = torch.argmax(max_logps[:, seq_len])

            while k > 0:
                best_path.insert(0, torch.tensor(k) - 1)
                prev_bp = bp
                bp = bps[bp, k]
                k = k - (prev_bp + 1)
            split_indices.append(best_path)

        return split_indices

    def inference(self,
        beam_ids,  # (batch_size * beam_size, tgt_cur_len)
        history_encodings,   # (batch_size * beam_size, tgt_cur_len, embed_dim)
        prev_seg_ends,  #  (batch_size * beam_size),
        src_token_ids,  # (batch_size * beam_siz, src_len)
        attn_scores  # (batch_size * beam_size, tgt_cur_len, src_len)
    ):

        num_beams = beam_ids.shape[0]
        seq_len = beam_ids.shape[1]
        cur_seg_lens = [seq_len - prev_seg_end for prev_seg_end in prev_seg_ends]
        prev_seg_ends = torch.tensor(prev_seg_ends).to(self.device)
        cur_seg_lens = torch.tensor(cur_seg_lens).to(self.device)
        char_vocab_size = self.char_decoder.vocab_size

        # Gather relevant history encoding for each sequence
        history_indices = prev_seg_ends.unsqueeze(-1)
        history_indices = history_indices.repeat(1, history_encodings.shape[-1])
        history_indices = history_indices.unsqueeze(1)
        history_encoding = torch.gather(history_encodings, dim=1, index=history_indices)
        history_encoding = history_encoding.squeeze(1)  # (batch_beam_size, embed_dim)

        attn_indices = prev_seg_ends.unsqueeze(-1)
        attn_indices = attn_indices.repeat(1, attn_scores.shape[-1])
        attn_indices = attn_indices.unsqueeze(1)
        attn_score = torch.gather(attn_scores, dim=1, index=attn_indices)

        lex_con_lprobs, lex_eom_lprobs = self.lex_inference(beam_ids, history_encoding, prev_seg_ends, cur_seg_lens,
                                                             char_vocab_size, src_token_ids, attn_score)

        # Extract history encoding up to last segment, prepare for all possible segments
        history_encoding = history_encoding.unsqueeze(0)  # (1, num_beams, embed_dim)
        history_encoding = torch.repeat_interleave(history_encoding, char_vocab_size, dim=1)  # (1, num_beams * char_vocab_size, embed_dim)

        # Collect and pad segment ids
        char_vocab_ids = torch.arange(char_vocab_size).to(self.device).unsqueeze(1)
        seg_ids = []
        seg_lens = []
        batch_lens = []
        for beam_num in range(num_beams):
            beam_seg_ids = beam_ids[beam_num, prev_seg_ends[beam_num]: ]
            seg_len = len(beam_seg_ids) + 1
            batch_lens.append(seg_len)
            seg_lens.extend([seg_len] * char_vocab_size)
            beam_seg_ids = beam_seg_ids.repeat(char_vocab_size, 1)
            beam_seg_ids = torch.cat([beam_seg_ids, char_vocab_ids], dim=1).transpose(0, 1)
            seg_ids.append(beam_seg_ids)
        padded_seg_ids = pad_sequence(seg_ids, padding_value=self.padding_idx)
        padded_seg_ids = padded_seg_ids.view(padded_seg_ids.shape[0], -1)

        # Extract input embeddings for current segments
        padded_seg_embeddings = self.embed_tokens(padded_seg_ids)  # (max_seg_len, batch_size * beam_size, embed_dim)
        packed_seg_embeddings = pack_padded_sequence(padded_seg_embeddings, torch.tensor(seg_lens), enforce_sorted=False)

        # Produce probabilities for segments including next characters and next characters+eom
        char_logits, _ = self.char_decoder(packed_seg_embeddings, history_encoding)
        char_lprobs = self.get_normalized_probs(net_output=(char_logits,), log_probs=True)  # (cur_seg_len+1, char_vocab_size, char_vocab_size)

        # Collect char probabilities
        char_vocab_lprobs = []
        eom_lprobs = []
        for beam_num in range(num_beams):
            beam_char_lprobs = char_lprobs[0: cur_seg_lens[beam_num] + 1,
                                            beam_num * char_vocab_size: beam_num * char_vocab_size + char_vocab_size, :]
            prev_char_lprob = 0.0
            if cur_seg_lens[beam_num] > 1:
                prev_char_lprobs = torch.gather(beam_char_lprobs[0: -2, 0], dim=-1,
                                                index=beam_ids[beam_num, prev_seg_ends[beam_num]+1:].unsqueeze(-1))
                prev_char_lprob = torch.sum(prev_char_lprobs)
            beam_char_vocab_lprobs = prev_char_lprob + beam_char_lprobs[-2, 0, :]  # add second last output to chain rule, for next character
            beam_eom_lprobs = beam_char_lprobs[-1, :, self.eom_id]  # last output for eom
            char_vocab_lprobs.append(beam_char_vocab_lprobs)
            eom_lprobs.append(beam_eom_lprobs)

        char_vocab_lprobs = torch.cat(char_vocab_lprobs, dim=0).view(num_beams, -1)
        eom_lprobs = torch.cat(eom_lprobs, dim=0).view(num_beams, -1)

        char_eom_lprobs = char_vocab_lprobs + eom_lprobs

        # Next character cannot be <pad>
        char_eom_lprobs[:, self.padding_idx] = -LOGINF
        char_vocab_lprobs[:, self.padding_idx] = -LOGINF

        # Next character cannot be eom itself
        char_eom_lprobs[:, self.eom_id] = -LOGINF
        char_vocab_lprobs[:, self.eom_id] = -LOGINF

        # Non-alphabetic character can only be one-character segment
        char_vocab_lprobs[:, 0: self.char_nspecial] = -LOGINF
        char_vocab_lprobs[:, self.char_nspecial + self.char_nalpha: ] = -LOGINF

        for beam_num in range(num_beams):
            # Only alphabetic characters can be part of multi-character segment
            if cur_seg_lens[beam_num] > 1:
                char_eom_lprobs[beam_num, 0: self.char_nspecial] = -LOGINF
                char_eom_lprobs[beam_num,self.char_nspecial + self.char_nalpha:] = -LOGINF

            # Segment cannot continue further if next character alread makes it as long as allowed
            if cur_seg_lens[beam_num] >= self.max_seg_len:
                char_vocab_lprobs[beam_num, :] = -LOGINF

        history_encoding = torch.gather(history_encodings, dim=1, index=history_indices)
        log_char_proportions, log_lex_proportions = self.compute_mix_lcoefs(
            history_encoding
        )

        char_con_element = log_char_proportions + char_vocab_lprobs
        lex_con_element = log_lex_proportions + lex_con_lprobs
        con_lprobs = torch.logsumexp(torch.stack([char_con_element, lex_con_element]), dim=0)

        char_eom_element = log_char_proportions + char_eom_lprobs
        lex_eom_element = log_lex_proportions + lex_eom_lprobs
        eom_lprobs = torch.logsumexp(torch.stack([char_eom_element, lex_eom_element]), dim=0)

        return con_lprobs, eom_lprobs

    def lex_inference(self,
        beam_ids,  # (batch_size * beam_size, tgt_cur_len)
        history_encoding,  # (batch_size * beam_size, tgt_cur_len, embed_dim)
        prev_seg_ends,  # (batch_size * beam_size)
        cur_seg_lens,   # (batch_size * beam_size)
        char_vocab_size,
        src_token_ids,  # (batch_size * beam_size, src_len)
        attn_scores  # (batch_size * beam_size, cur_seq_len + 1, src_len)
    ):

        num_beams = beam_ids.shape[0]
        src_len = src_token_ids.shape[1]
        log_attn_scores = torch.log(attn_scores + 1e-10)

        # Compute lexicon lprobs
        gen_logits = self.lex_decoder(history_encoding)  # (batch_size * beam_size, lex_vocab_size)
        full_gen_logp = self.get_normalized_probs(net_output=(gen_logits,),
                                                  log_probs=True)  # in models.fairseq_decoder  # (lex_vocab_size)
        full_gen_logp = torch.cat((full_gen_logp, torch.full((full_gen_logp.shape[0], 1), fill_value=-LOGINF,
                                                             device=self.device)), dim=-1)

        # Prepare indices
        target_ids = beam_ids.transpose(0, 1)  # (tgt_cur_len, batch_size * beam_size)
        target_ids = torch.repeat_interleave(target_ids, char_vocab_size, dim=1)  # (tgt_cur_len, batch_size * beam_size * char_vocab_size)
        char_vocab_ids = torch.arange(char_vocab_size).repeat(num_beams).unsqueeze(0).to(self.device)
        target_ids = torch.cat([target_ids, char_vocab_ids], dim=0)
        seg_starts = [prev_seg_end + 1 for prev_seg_end in prev_seg_ends]

        seg_char_ids = []
        for beam_num in range(num_beams):
            beam_seg_char_ids = target_ids[seg_starts[beam_num]: seg_starts[beam_num] + cur_seg_lens[beam_num],
                                    beam_num * char_vocab_size: beam_num * char_vocab_size + char_vocab_size].T.tolist()
            beam_seg_char_ids = [tuple(char_ids) for char_ids in beam_seg_char_ids]
            seg_char_ids.extend(beam_seg_char_ids)

        # Collect ended segment lprobs
        seg_lex_ids = torch.LongTensor([self.chars2lex[char_ids] if char_ids in self.chars2lex
                                        else self.lex_vocab_size  # not in segment lexicon
                                        for char_ids in seg_char_ids]).to(self.device)
        seg_lex_ids = seg_lex_ids.view(num_beams, -1)
        lex_eom_logp = torch.gather(full_gen_logp, dim=-1, index=seg_lex_ids).squeeze(-1)

        # Collect continued segment lprobs
        preseg_lex_ids = [self.prechars2lex[char_ids] if char_ids in self.prechars2lex
                          else [self.lex_vocab_size]  # not in pre-segment lexicon
                          for char_ids in seg_char_ids]
        max_lex_mappings = max([len(lex_ids) for lex_ids in preseg_lex_ids])
        for i in range(len(preseg_lex_ids)):
            num_lex_mappings = len(preseg_lex_ids[i])
            preseg_lex_ids[i].extend([self.lex_vocab_size] * (max_lex_mappings - num_lex_mappings))
        preseg_lex_ids = torch.tensor(preseg_lex_ids).view(num_beams, char_vocab_size * max_lex_mappings).to(self.device)
        lex_con_logp = torch.gather(full_gen_logp, dim=-1, index=preseg_lex_ids).view(num_beams, char_vocab_size, max_lex_mappings)
        lex_con_logp = torch.logsumexp(lex_con_logp, dim=-1)

        if self.copy:

            # Combine copy and gen coefficients
            log_gen_proportions, log_copy_proportions = self.compute_pointer_lcoefs(
                history_encoding
            )

            """EOM"""
            # Map src tokens to lexicon ids
            src_lex_ids = torch.full(src_token_ids.shape, fill_value=-1, dtype=torch.long, device=self.device)
            for batch_num in range(num_beams):
                for src_pos in range(src_len):
                    if src_token_ids[batch_num][src_pos].item() in self.tokens2lex:
                        src_lex_ids[batch_num][src_pos] = self.tokens2lex[src_token_ids[batch_num][src_pos].item()]

            # Extract copy probabilities for lexicon items
            full_copy_lex_logp = torch.full_like(full_gen_logp, fill_value=-LOGINF, device=self.device)  # (seq_len, batch_size, lex_vocab_size)
            full_copy_lex_p = torch.full_like(full_gen_logp, fill_value=0, device=self.device)
            for batch_num in range(num_beams):
                for src_pos in range(src_len):
                    if src_lex_ids[batch_num][src_pos] != -1:
                        full_copy_lex_logp[batch_num, src_lex_ids[batch_num][src_pos]] = \
                            torch.logaddexp(full_copy_lex_logp[batch_num, src_lex_ids[batch_num][src_pos]].clone(),
                                            log_attn_scores[batch_num, -1, src_pos])
                        full_copy_lex_p[batch_num, src_lex_ids[batch_num][src_pos]] += \
                                            attn_scores[batch_num, -1, src_pos]

            # Extract copy probabilities for out-of-lexicon items
            full_copy_unk_logp = torch.full((num_beams, self.src_vocab_size),
                                            fill_value=-LOGINF, device=self.device)
            full_copy_unk_logp = torch.cat((full_copy_unk_logp, torch.full((full_copy_unk_logp.shape[0], 1),
                                                                 fill_value=-LOGINF, device=self.device)), dim=-1)

            for batch_num in range(num_beams):
                for src_pos in range(src_len):
                    if src_lex_ids[batch_num][src_pos] == -1 and src_token_ids[batch_num][src_pos] >= self.src_dictionary.nspecial:  # Check if unk
                        full_copy_unk_logp[batch_num, src_token_ids[batch_num][src_pos]] = \
                            torch.logaddexp(full_copy_unk_logp[batch_num, src_token_ids[batch_num][src_pos]].clone(),
                                            log_attn_scores[batch_num, -1, src_pos])

            # Extract lexicon copy lprobs
            copy_lex_eom_logp = torch.gather(full_copy_lex_logp, dim=-1,
                                             index=seg_lex_ids).squeeze(-1)

            # Get source ids for any segments not in lexicon
            seg_src_ids = torch.LongTensor([self.chars2tokens[char_ids] if char_ids in self.chars2tokens
                                                                           and char_ids not in self.chars2lex
                                            else self.src_vocab_size  # not in segment lexicon
                                            for char_ids in seg_char_ids]).to(self.device)
            seg_src_ids = seg_src_ids.view(num_beams, -1)

            copy_unk_eom_logp = torch.gather(full_copy_unk_logp, dim=-1,
                                         index=seg_src_ids).squeeze(-1)

            # Combine lex and unk logps
            copy_eom_logp = torch.logsumexp(torch.stack([copy_lex_eom_logp, copy_unk_eom_logp]), dim=0)


            gen_element = log_gen_proportions.unsqueeze(-1) + lex_eom_logp
            copy_element = log_copy_proportions.unsqueeze(-1) + copy_eom_logp
            lex_eom_logp = torch.logsumexp(torch.stack([gen_element, copy_element]), dim=0)

            """CON"""
            # Get possible future source token ids for char segments
            preseg_src_ids = [self.prechars2tokens[char_ids] if char_ids in self.prechars2tokens
                                               else [self.src_vocab_size]  # not in segment lexicon
                                               for char_ids in seg_char_ids]

            # Map src tokens as prefixes to potential lexicon ids
            preseg_lex_ids = []
            for token_ids in preseg_src_ids:
                preseg_lex_ids.append([self.tokens2lex[token_id] if token_id in self.tokens2lex
                                       else self.lex_vocab_size  # not in pre-segment lexicon
                                       for token_id in token_ids])
            new_preseg_src_ids = []
            for i, lex_ids in enumerate(preseg_lex_ids):
                new_preseg_src_ids.append([])
                for j, lex_id in enumerate(lex_ids):
                    if lex_id == self.lex_vocab_size:
                        new_preseg_src_ids[i].append(preseg_src_ids[i][j])
                if len(new_preseg_src_ids[i]) == 0:
                    new_preseg_src_ids[i].append(self.src_vocab_size)
            preseg_src_ids = new_preseg_src_ids

            max_src_mappings = max([len(src_ids) for src_ids in preseg_src_ids])
            for i in range(len(preseg_src_ids)):
                num_src_mappings = len(preseg_src_ids[i])
                preseg_src_ids[i].extend([self.src_vocab_size] * (max_src_mappings - num_src_mappings))
            preseg_src_ids = torch.tensor(preseg_src_ids).view(num_beams, char_vocab_size * max_src_mappings).to(
                self.device)

            max_lex_mappings = max([len(lex_ids) for lex_ids in preseg_lex_ids])
            for i in range(len(preseg_lex_ids)):
                num_lex_mappings = len(preseg_lex_ids[i])
                preseg_lex_ids[i].extend([self.lex_vocab_size] * (max_lex_mappings - num_lex_mappings))
            preseg_lex_ids = torch.tensor(preseg_lex_ids).view(num_beams, char_vocab_size * max_lex_mappings).to(
                self.device)

            # Extract copy probabilities for lexicon items
            copy_lex_con_logp = torch.gather(full_copy_lex_logp, dim=-1, index=preseg_lex_ids).view(num_beams, char_vocab_size,
                                                                                          max_lex_mappings)
            copy_lex_con_logp = torch.logsumexp(copy_lex_con_logp, dim=-1)

            # Extract copy probabilities for unk items
            copy_unk_con_logp = torch.gather(full_copy_unk_logp, dim=-1, index=preseg_src_ids).view(num_beams, char_vocab_size,
                                                                                            max_src_mappings)
            copy_unk_con_logp = torch.logsumexp(copy_unk_con_logp, dim=-1)

            # Combine lex and unk logps
            copy_con_logp = torch.logsumexp(torch.stack([copy_lex_con_logp, copy_unk_con_logp]), dim=0)
            gen_element = log_gen_proportions.unsqueeze(-1) + lex_con_logp
            copy_element = log_copy_proportions.unsqueeze(-1) + copy_con_logp
            lex_con_logp = torch.logsumexp(torch.stack([gen_element, copy_element]), dim=0)

        return lex_con_logp, lex_eom_logp


    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert (
            srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores: Optional[Tensor] = (
            x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        )
        outs = []
        hiddens = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                out, attn_scores[:, j, :] = self.attention(
                    hidden, encoder_outs, encoder_padding_mask
                )
            else:
                out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)
            hiddens.append(hidden)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        h = torch.cat(hiddens, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        h = h.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores, h

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def copy_output_layer(
        self,
        features: Tensor,
        attn: Tensor,
        src_tokens: Tensor,
        p_gen_logits: Tensor,
    ) -> Tensor:
        """
        Project features to the vocabulary size and mix with the attention
        distributions.
        """
        logcoef_gen = F.logsigmoid(p_gen_logits)
        logcoef_copy = F.logsigmoid(-p_gen_logits)

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            logits = self.output_layer(features)
        else:
            logits = features

        batch_size = logits.shape[0]
        output_length = logits.shape[1]
        assert logits.shape[2] == self.num_char_embeddings
        assert src_tokens.shape[0] == batch_size
        src_length = src_tokens.shape[1]

        # The final output distribution will be a mixture of the normal output
        # distribution (softmax of logits) and attention weights.
        logp_gen = self.get_normalized_probs_scriptable(
            (logits, None), log_probs=True, sample=None
        )
        logp_gen = logcoef_gen + logp_gen
        padding_size = (batch_size, output_length, self.num_oov_types)
        padding = torch.full(padding_size, fill_value=-LOGINF)
        logp_gen = torch.cat((logp_gen, padding), 2)
        assert logp_gen.shape[2] == self.num_types

        # Scatter attention distributions to distributions over the extended
        # vocabulary in a tensor of shape [batch_size, output_length,
        # vocab_size]. Each attention weight will be written into a location
        # that is for other dimensions the same as in the index tensor, but for
        # the third dimension it's the value of the index tensor (the token ID).
        log_attn = torch.log(attn)  # torch.log(attn.float() + 1e-20
        log_attn = logcoef_copy + log_attn
        index = src_tokens[:, None, :]
        index = index.expand(batch_size, output_length, src_length)
        attn_dists_size = (batch_size, output_length, self.num_types)
        log_attn_dists = torch.full(attn_dists_size, fill_value=-LOGINF)
        log_attn_dists.scatter_(2, index, log_attn.float())

        # Final distributions, [batch_size, output_length, num_types].
        return torch.logsumexp(torch.stack([logp_gen, log_attn_dists]), dim=0)

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        input_feed = cached_state[
            "input_feed"
        ]  # can be None for decoder-only language models
        return prev_hiddens, prev_cells, input_feed

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn



class LexDecoder(nn.Module):
    """
    Once-off lexical generation of a segment, conditioned on the sequence history.
    """
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(LexDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.drop(hidden_states)
        logits = self.fc(hidden_states)
        return logits


class LSTMCharDecoder(nn.Module):
    """
    Character by character generation of a segment, conditioned on the sequence history.
    """
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, dropout):
        super(LSTMCharDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.drop = nn.Dropout(dropout)  # dropout used for embedding and final layer
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, embedding, init_hidden_states):
        """
        :param embedding: (seg_len + 1, num_segs, embedding_dim)
        :param init_hidden_states: (1, num_segs, embedding_dim)
        :return: logits: (seg_len + 1, num_segs, vocab_size)
        """
        init_hidden_states = self.transform(init_hidden_states)
        init_cell_states = torch.zeros_like(init_hidden_states)
        hidden_states, final_states = self.lstm(embedding, (init_hidden_states, init_cell_states))
        if type(hidden_states) is PackedSequence:
            hidden_states, _ = pad_packed_sequence(hidden_states)
        output = self.drop(hidden_states)
        logits = self.fc(output)
        return logits, (final_states[0].detach(), final_states[1].detach())

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class CopyEmbedding(nn.Embedding):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings. This subclass differs from the standard PyTorch Embedding class by
    allowing additional vocabulary entries that will be mapped to the unknown token
    embedding.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int): Pads the output with the embedding vector at :attr:`padding_idx`
                           (initialized to zeros) whenever it encounters the index.
        unk_idx (int): Maps all token indices that are greater than or equal to
                       num_embeddings to this index.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`
    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)
    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.
    """
    __constants__ = ["unk_idx"]

    # Torchscript: Inheriting from Embedding class produces an error when exporting to Torchscript
    # -> RuntimeError: Unable to cast Python instance to C++ type (compile in debug mode for details
    # It's happening because max_norm attribute from nn.Embedding is None by default and it cannot be
    # cast to a C++ type
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int],
        unk_idx: int,
        max_norm: Optional[float] = float("inf"),
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm)
        self.unk_idx = unk_idx

        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.constant_(self.weight[padding_idx], 0)

    def forward(self, input):
        input = torch.where(
            input >= self.num_embeddings, torch.ones_like(input) * self.unk_idx, input
        )
        return nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture("ssd2t", "ssd2t")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", False)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_attention = getattr(args, "decoder_attention", "1")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )

    # Custom arguments
    args.decoder_copy = getattr(args, "decoder_copy", "0")


@register_model_architecture("ssd2t", "ssd2t_wiseman_iwslt_de_en")
def lstm_wiseman_iwslt_de_en(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", 0)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", 0)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 256)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", 0)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    base_architecture(args)


@register_model_architecture("ssd2t", "ssd2t_luong_wmt_en_de")
def lstm_luong_wmt_en_de(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1000)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", 0)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1000)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1000)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", 0)
    base_architecture(args)
