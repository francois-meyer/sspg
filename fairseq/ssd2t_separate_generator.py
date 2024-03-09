# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

def generate(beam_ids, dict, eoms):

    text = "".join([dict.symbols[id] for id in beam_ids[0, 1:]])

    for counter, index in enumerate(eoms):
        if index == 0:
            continue
        text = text[:index + counter - 1] + "-" + text[index + counter - 1: ]

    print(text)


class SubwordSegmentalSeparateGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        average_next_scores=False,
        normalize_type=None,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len or self.model.max_decoder_positions()

        self.normalize_scores = normalize_scores
        self.average_next_scores = average_next_scores
        self.normalize_type = normalize_type
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(
        self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, 1).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token

        # number of candidate hypos per step
        cand_size = 3 * beam_size  # 2 x beam size in case half are EOS

        reorder_state: Optional[Tensor] = None

        beam_ids = []  # continued segments
        beam_lprobs = []
        beam_eoms = [[[0] for _ in range(beam_size)] for _ in range(bsz)]

        active_batch_indices = [i for i in range(bsz)]
        active_bsz = bsz
        to_be_finalized = [beam_size for _ in range(bsz)]  # number of sentences remaining
        finalized_sent_ids = [[] for _ in range(bsz)]  # completed
        finalized_sent_lprobs = [[] for _ in range(bsz)]
        finalized_sent_eoms = [[] for _ in range(bsz)]

        decode_normalization_type, final_normalization_type = self.normalize_type.split("-")

        LOGINF = math.inf
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lex2chars = self.model.single_model.decoder.lex2chars
        tokens2chars = self.model.single_model.decoder.tokens2chars
        tokens2chars[0] = (0,)
        tokens2chars[1] = (1,)
        tokens2chars[2] = (2,)
        tokens2chars[3] = (3,)
        self.model.single_model.decoder.decoding = "separate"
        for step in range(max_len + 1):  # 0 is first letter, one extra step for EOS marker

            with torch.autograd.profiler.record_function(
                "EnsembleModel: forward_decoder"
            ):
                if step == 0:
                    prev_seg_end = [0] * bsz
                    init_beam_ids = torch.tensor([[self.eos]] * bsz)
                    prev_output_tokens = (init_beam_ids, prev_seg_end)

                    next_lex_lprobs, next_src_lprobs, top_char_segs, top_char_lprobs, history_encodings, attn_scores = self.model.forward_decoder(
                        prev_output_tokens,
                        encoder_outs,
                        incremental_states,
                        self.temperature,
                    )

                    # Handle min length constraint
                    next_lex_lprobs[:, self.eos] = -LOGINF
                    next_src_lprobs[:, self.eos] = -LOGINF

                    for batch_num in range(bsz):
                        batch_lex_lprobs = next_lex_lprobs[batch_num]
                        batch_src_lprobs = next_src_lprobs[batch_num]
                        beam_ids.append([])
                        beam_lprobs.append([])

                        next_char_ids = []
                        next_seg_lprobs = []
                        for beam_num, index in enumerate(torch.topk(batch_lex_lprobs, k=beam_size).indices):
                            char_ids = list(lex2chars[index.item()])
                            next_char_ids.append(char_ids)
                            next_seg_lprobs.append(batch_lex_lprobs[index])

                        for beam_num, index in enumerate(torch.topk(batch_src_lprobs, k=beam_size).indices):
                            if index.item() in tokens2chars:
                                char_ids = list(tokens2chars[index.item()])
                                next_char_ids.append(char_ids)
                                next_seg_lprobs.append(batch_src_lprobs[index])
                            else:
                                char_ids = [self.model.single_model.decoder.tgt_dictionary.unk_index]
                                next_char_ids.append(char_ids)
                                next_seg_lprobs.append(batch_src_lprobs[index])

                        for i, char_ids in enumerate(top_char_segs[batch_num]):
                            next_char_ids.append(char_ids)
                            next_seg_lprobs.append(top_char_lprobs[batch_num, i])

                        next_seg_lprobs = torch.tensor(next_seg_lprobs)

                        for beam_num, index in enumerate(torch.topk(next_seg_lprobs, k=beam_size).indices):
                            char_ids = next_char_ids[index]
                            beam_ids[batch_num].append(torch.tensor([[self.eos] + char_ids]))
                            beam_lprobs[batch_num].append(torch.tensor([next_seg_lprobs[index].unsqueeze(-1)]).to(device))
                            beam_eoms[batch_num][beam_num].append(len(char_ids))

                    # Prepare encoder outputs for multiple beams per batch
                    encoder_outs[0] = list(encoder_outs[0])
                    encoder_outs[0][0] = torch.repeat_interleave(  # (src_len, bsz * beam_size, -1)
                        encoder_outs[0][0], beam_size, dim=1).repeat(1, 1, 1)

                    encoder_outs[0][1] = torch.repeat_interleave(  # (1, bsz * beam_size, -1)
                        encoder_outs[0][1], beam_size, dim=1).repeat(1, 1, 1)

                    encoder_outs[0][2] = torch.repeat_interleave(  # (1, bsz * beam_size, -1)
                        encoder_outs[0][2], beam_size, dim=1).repeat(1, 1, 1)

                    encoder_outs[0][3] = torch.repeat_interleave(  # (src_len, bsz * beam_size)
                        encoder_outs[0][3], beam_size, dim=1).repeat(1, 1)

                    encoder_outs[0][4] = torch.repeat_interleave(  # (bsz * beam_size, src_len)
                        encoder_outs[0][4], beam_size, dim=0).repeat(1, 1)

                    # Prepare incremental states for multiple beams per batch
                    reorder_state = torch.tensor([[batch_num] * beam_size for batch_num in range(bsz)]).flatten()
                    reorder_state = reorder_state.to(device)
                else:

                    if step >= max_len:
                        print("Reached max step without finalizing all sentences.")

                    # Prep con_beams
                    prev_ids = [ids.transpose(0, 1) for batch_ids in beam_ids for ids in batch_ids]
                    prev_ids = pad_sequence(prev_ids, padding_value=self.pad).transpose(0, 1).squeeze(-1)
                    prev_eoms = [beam_eoms[batch_num][beam_num][-1] for batch_num in range(active_bsz)
                                     for beam_num in range(beam_size)]

                    prev_output_tokens = (prev_ids, prev_eoms, None, None)
                    self.model.reorder_incremental_state(incremental_states, reorder_state)

                    next_lex_lprobs, next_src_lprobs, top_char_segs, top_char_lprobs, history_encodings, attn_scores = self.model.forward_decoder(
                        prev_output_tokens,
                        encoder_outs,
                        incremental_states,
                        self.temperature,
                    )

                    # Handle max length constraint
                    if step == max_len:  # Force eos
                        next_lex_lprobs[:, :] = -LOGINF
                        next_lex_lprobs[:, 0: self.eos] = -LOGINF
                        next_lex_lprobs[:, self.eos + 1:] = -LOGINF

                        next_src_lprobs[:, :] = -LOGINF
                        next_src_lprobs[:, 0: self.eos] = -LOGINF
                        next_src_lprobs[:, self.eos + 1:] = -LOGINF
                        next_src_lprobs[:, self.eos] = torch.log(torch.tensor(0.99))

                    elif step < self.min_len:
                        next_lex_lprobs[:, self.eos] = -LOGINF
                        next_src_lprobs[:, self.eos] = -LOGINF

                    next_lex_lprobs = next_lex_lprobs.view(active_bsz, beam_size, -1)
                    next_src_lprobs = next_src_lprobs.view(active_bsz, beam_size, -1)

                    new_top_char_segs = []
                    k = 0
                    for i in range(active_bsz):
                        new_top_char_segs.append([])
                        for j in range(beam_size):
                            new_top_char_segs[i].append(top_char_segs[k])
                            k += 1
                    top_char_segs = new_top_char_segs

                    top_char_lprobs = top_char_lprobs.view(active_bsz, beam_size, -1)

                    max_seg_ids = []  # stores char ids
                    max_next_seg_lprobs = []
                    max_seg_lprobs = []

                    for batch_num in range(active_bsz):
                        max_seg_ids.append([])
                        max_next_seg_lprobs.append([])
                        max_seg_lprobs.append([])

                        # for beam_num, prev_ids in enumerate(con_beam_ids):
                        for beam_num in range(beam_size):
                            beam_next_lex_lprobs = next_lex_lprobs[batch_num, beam_num]
                            for index in torch.topk(beam_next_lex_lprobs, k=beam_size).indices:
                                max_seg_ids[batch_num].append(list(lex2chars[index.item()]))
                                max_next_seg_lprobs[batch_num].append(beam_next_lex_lprobs[index])
                                max_seg_lprobs[batch_num].append(torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) +
                                                                 beam_next_lex_lprobs[index])
                                if decode_normalization_type == "seg":
                                    max_seg_lprobs[batch_num][-1] /= (len(beam_eoms[batch_num][beam_num]))
                                elif decode_normalization_type == "char":
                                    max_seg_lprobs[batch_num][-1] /= (step + len(beam_eoms[batch_num][beam_num]))

                            beam_next_src_lprobs = next_src_lprobs[batch_num, beam_num]
                            for index in torch.topk(beam_next_src_lprobs, k=beam_size).indices:
                                if index.item() in tokens2chars:
                                    max_seg_ids[batch_num].append(list(tokens2chars[index.item()]))
                                    max_next_seg_lprobs[batch_num].append(beam_next_src_lprobs[index])
                                    max_seg_lprobs[batch_num].append(torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) +
                                                                        beam_next_src_lprobs[index])
                                else:
                                    max_seg_ids[batch_num].append([self.model.single_model.decoder.tgt_dictionary.unk_index])
                                    max_next_seg_lprobs[batch_num].append(beam_next_src_lprobs[index])
                                    max_seg_lprobs[batch_num].append(torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) +
                                                                        beam_next_src_lprobs[index])
                                if decode_normalization_type == "seg":
                                    max_seg_lprobs[batch_num][-1] /= (len(beam_eoms[batch_num][beam_num]))
                                elif decode_normalization_type == "char":
                                    max_seg_lprobs[batch_num][-1] /= (step + len(beam_eoms[batch_num][beam_num]))

                            beam_next_char_lprobs = top_char_lprobs[batch_num, beam_num]
                            for i, lprob in enumerate(beam_next_char_lprobs):
                                char_ids = top_char_segs[batch_num][beam_num][i]
                                max_seg_ids[batch_num].append(char_ids)
                                max_next_seg_lprobs[batch_num].append(beam_next_char_lprobs[i])
                                max_seg_lprobs[batch_num].append(torch.sum(beam_lprobs[batch_num][beam_num][0: -1]) +
                                                             beam_next_char_lprobs[i])
                                if decode_normalization_type == "seg":
                                    max_seg_lprobs[batch_num][-1] /= (len(beam_eoms[batch_num][beam_num]))
                                elif decode_normalization_type == "char":
                                    max_seg_lprobs[batch_num][-1] /= (step + len(beam_eoms[batch_num][beam_num]))

                    # Compare new continued segments
                    new_beam_ids = []
                    new_beam_lprobs = []
                    new_beam_eoms = []
                    new_history_embeddings = []
                    new_reorder_states = []

                    for batch_num in range(active_bsz):
                        new_beam_ids.append([])
                        new_beam_lprobs.append([])
                        new_beam_eoms.append([])
                        new_history_embeddings.append([])
                        new_reorder_states.append([])

                        for beam_num in range(cand_size):
                            max_lprob = max(max_seg_lprobs[batch_num])
                            max_index = max_seg_lprobs[batch_num].index(max_lprob)
                            max_beam_num = int(max_index / (beam_size * 3))
                            max_id = max_seg_ids[batch_num][max_index]

                            beam_max_ids = beam_ids[batch_num][max_beam_num]
                            new_char_ids = max_id
                            new_beam_ids[batch_num].append(torch.cat([beam_max_ids, torch.tensor([new_char_ids])], dim=-1))

                            beam_max_lprobs = beam_lprobs[batch_num][max_beam_num].detach().clone()
                            beam_max_lprobs = torch.cat(
                                [beam_max_lprobs, torch.tensor([max_seg_lprobs[batch_num][max_index]]).to(device)])
                            new_beam_lprobs[batch_num].append(beam_max_lprobs)

                            beam_max_eoms = beam_eoms[batch_num][max_beam_num].copy()  # necessary?
                            beam_max_eoms.append(beam_max_eoms[-1] + len(new_char_ids))
                            new_beam_eoms[batch_num].append(beam_max_eoms)

                            max_seg_lprobs[batch_num][max_index] = -LOGINF
                            new_reorder_states[batch_num].append(
                                active_bsz * beam_size + batch_num * beam_size + max_beam_num)

                            for cand_num in range(beam_size * (beam_size * 3)):
                                cand_beam_num = int(cand_num / (beam_size * 3))
                                if torch.equal(beam_max_ids, beam_ids[batch_num][cand_beam_num]) and\
                                        new_char_ids == max_seg_ids[batch_num][cand_num] and\
                                        beam_eoms[batch_num][max_beam_num] == beam_eoms[batch_num][cand_beam_num]:
                                    max_seg_lprobs[batch_num][cand_num] = -LOGINF

                    # Collect top beam_size beams and store finished sentences
                    beam_ids = []
                    beam_lprobs = []
                    beam_eoms = []
                    history_embeddings = []
                    reorder_states = []

                    deactivated_batch_nums = []
                    deactivated_batch_indices = []
                    for batch_num in range(active_bsz):
                        beam_ids.append([])
                        beam_lprobs.append([])
                        beam_eoms.append([])
                        history_embeddings.append([])
                        reorder_states.append([])

                        cand_num = 0
                        while len(beam_ids[batch_num]) < beam_size:

                            if new_beam_ids[batch_num][cand_num][0, -1] != self.eos:
                                beam_ids[batch_num].append(new_beam_ids[batch_num][cand_num])
                                beam_lprobs[batch_num].append(new_beam_lprobs[batch_num][cand_num])
                                beam_eoms[batch_num].append(new_beam_eoms[batch_num][cand_num])
                                reorder_states[batch_num].append(new_reorder_states[batch_num][cand_num])
                            elif new_beam_ids[batch_num][cand_num][0, -1] == self.eos:
                                batch_index = active_batch_indices[batch_num]
                                finalized_sent_ids[batch_index].append(new_beam_ids[batch_num][cand_num])
                                finalized_sent_lprobs[batch_index].append(new_beam_lprobs[batch_num][cand_num])
                                finalized_sent_eoms[batch_index].append(new_beam_eoms[batch_num][cand_num])
                                to_be_finalized[batch_index] -= 1
                                if to_be_finalized[batch_index] == 0:
                                    active_bsz -= 1
                                    deactivated_batch_indices.append(batch_index)
                                    deactivated_batch_nums.append(batch_num)
                                    break
                            cand_num += 1

                    if step == max_len or active_bsz == 0:
                        break

                    for batch_index in deactivated_batch_indices:
                        active_batch_indices.remove(batch_index)

                    for batch_num in sorted(deactivated_batch_nums, reverse=True):
                        del beam_ids[batch_num]
                        del beam_lprobs[batch_num]
                        del beam_eoms[batch_num]
                        del history_embeddings[batch_num]
                        del reorder_states[batch_num]

                        # Discard encodings of sentence _end source sentence
                        start = batch_num * beam_size
                        end = batch_num * beam_size + beam_size
                        encoder_outs[0][0] = torch.cat(
                            [encoder_outs[0][0][:, 0: start],
                             encoder_outs[0][0][:, end:]], dim=1)

                        encoder_outs[0][1] = torch.cat(
                            [encoder_outs[0][1][:, 0: start],
                             encoder_outs[0][1][:, end:]], dim=1)

                        encoder_outs[0][2] = torch.cat(
                            [encoder_outs[0][2][:, 0: start],
                             encoder_outs[0][2][:, end:]], dim=1)

                        encoder_outs[0][3] = torch.cat(
                            [encoder_outs[0][3][:, 0: start],
                             encoder_outs[0][3][:, end:]], dim=1)

                        encoder_outs[0][4] = torch.cat(
                            [encoder_outs[0][4][0: start],
                             encoder_outs[0][4][end:]], dim=0)

        finalized = []
        for batch_num in range(bsz):
            finalized.append([])
            for beam_num in range(beam_size):
                normalizer = 1
                if self.normalize_scores:
                    if final_normalization_type == "seg":
                        normalizer = len(finalized_sent_eoms[batch_num][beam_num]) - 1
                    elif final_normalization_type == "char":
                        normalizer = len(finalized_sent_eoms[batch_num][beam_num]) - 1 + len(finalized_sent_ids[batch_num][beam_num])
                score = torch.sum(finalized_sent_lprobs[batch_num][beam_num]) / normalizer

                finalized[batch_num].append({"tokens": finalized_sent_ids[batch_num][beam_num],
                                         "score": score,
                                         "eoms": finalized_sent_eoms[batch_num][beam_num],
                                         "alignment": None,
                                         "positional_scores": finalized_sent_lprobs[batch_num][beam_num]})

        # sort by score descending
        for batch_num in range(bsz):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[batch_num]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[batch_num] = [finalized[batch_num][ssi] for ssi in sorted_scores_indices]
            finalized[batch_num] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[batch_num]
            )
        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))


    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min(
            [
                m.max_decoder_positions()
                for m in self.models
                if hasattr(m, "max_decoder_positions")
            ]
            + [sys.maxsize]
        )

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)

        return decoder_out

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


