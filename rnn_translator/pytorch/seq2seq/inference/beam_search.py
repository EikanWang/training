import torch

from mlperf_compliance import mlperf_log

from seq2seq.data.config import BOS
from seq2seq.data.config import EOS
import numpy as np

import copy

class SequenceGenerator(object):
    def __init__(self,
                 model,
                 beam_size=5,
                 max_seq_len=100,
                 cuda=False,
                 len_norm_factor=0.6,
                 len_norm_const=5,
                 cov_penalty_factor=0.1):

        self.model = model
        self.cuda = cuda
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.len_norm_factor = len_norm_factor
        self.len_norm_const = len_norm_const
        self.cov_penalty_factor = cov_penalty_factor

        self.batch_first = self.model.batch_first

        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_BEAM_SIZE,
                              value=self.beam_size)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_MAX_SEQ_LEN,
                              value=self.max_seq_len)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_LEN_NORM_CONST,
                              value=self.len_norm_const)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_LEN_NORM_FACTOR,
                              value=self.len_norm_factor)
        mlperf_log.gnmt_print(key=mlperf_log.EVAL_HP_COV_PENALTY_FACTOR,
                              value=self.cov_penalty_factor)

    def greedy_search(self, batch_size, initial_input, initial_context=None):
        max_seq_len = self.max_seq_len

        translation = torch.zeros(batch_size, max_seq_len, dtype=torch.int64)
        lengths = torch.ones(batch_size, dtype=torch.int64)
        active = torch.arange(0, batch_size, dtype=torch.int64)
        base_mask = torch.arange(0, batch_size, dtype=torch.int64)

        if self.cuda:
            translation = translation.cuda()
            lengths = lengths.cuda()
            active = active.cuda()
            base_mask = base_mask.cuda()

        translation[:, 0] = BOS
        words, context = initial_input, initial_context

        if self.batch_first:
            word_view = (-1, 1)
            ctx_batch_dim = 0
        else:
            word_view = (1, -1)
            ctx_batch_dim = 1

        counter = 0
        for idx in range(1, max_seq_len):
            if not len(active):
                break
            counter += 1

            words = words.view(word_view)
            words, logprobs, attn, context = self.model.generate(words, context, 1)
            words = words.view(-1)

            translation[active, idx] = words
            lengths[active] += 1

            terminating = (words == EOS)

            if terminating.any():
                not_terminating = ~terminating

                mask = base_mask[:len(active)]
                mask = mask.masked_select(not_terminating)
                active = active.masked_select(not_terminating)

                words = words[mask]
                context[0] = context[0].index_select(ctx_batch_dim, mask)
                context[1] = context[1].index_select(0, mask)
                context[2] = context[2].index_select(1, mask)

        return translation, lengths, counter

    def beam_search(self, batch_size, initial_input, initial_context=None):
        beam_size = self.beam_size
        norm_const = self.len_norm_const
        norm_factor = self.len_norm_factor
        max_seq_len = self.max_seq_len
        cov_penalty_factor = self.cov_penalty_factor

        translation = torch.zeros(batch_size * beam_size, max_seq_len, dtype=torch.int64)
        translation_bf16 = copy.deepcopy(translation)
        lengths = torch.ones(batch_size * beam_size, dtype=torch.int64)
        lengths_bf16 = copy.deepcopy(lengths)
        scores = torch.zeros(batch_size * beam_size, dtype=torch.float32)
        scores_bf16 = copy.deepcopy(scores)

        active = torch.arange(0, batch_size * beam_size, dtype=torch.int64)
        active_bf16 = copy.deepcopy(active)
        base_mask = torch.arange(0, batch_size * beam_size, dtype=torch.int64)
        global_offset = torch.arange(0, batch_size * beam_size, beam_size, dtype=torch.int64)

        eos_beam_fill = torch.tensor([0] + (beam_size - 1) * [float('-inf')])

        if self.cuda:
            translation = translation.cuda()
            lengths = lengths.cuda()
            active = active.cuda()
            base_mask = base_mask.cuda()
            scores = scores.cuda()
            global_offset = global_offset.cuda()
            eos_beam_fill = eos_beam_fill.cuda()

        translation[:, 0] = BOS
        translation_bf16[:, 0] = BOS

        words, context = initial_input, initial_context

        if self.batch_first:
            word_view = (-1, 1)
            ctx_batch_dim = 0
            attn_query_dim = 1
        else:
            word_view = (1, -1)
            ctx_batch_dim = 1
            attn_query_dim = 0

        # replicate context
        if self.batch_first:
            # context[0] (encoder state): (batch, seq, feature)
            _, seq, feature = context[0].shape
            context[0].unsqueeze_(1)
            context[0] = context[0].expand(-1, beam_size, -1, -1)
            context[0] = context[0].contiguous().view(batch_size * beam_size, seq, feature)
            # context[0]: (batch * beam, seq, feature)
        else:
            # context[0] (encoder state): (seq, batch, feature)
            seq, _, feature = context[0].shape
            context[0].unsqueeze_(2)
            context[0] = context[0].expand(-1, -1, beam_size, -1)
            context[0] = context[0].contiguous().view(seq, batch_size * beam_size, feature)
            # context[0]: (seq, batch * beam,  feature)

        #context[1] (encoder seq length): (batch)
        context[1].unsqueeze_(1)
        context[1] = context[1].expand(-1, beam_size)
        context[1] = context[1].contiguous().view(batch_size * beam_size)
        #context[1]: (batch * beam)

        accu_attn_scores = torch.zeros(batch_size * beam_size, seq)
        if self.cuda:
            accu_attn_scores = accu_attn_scores.cuda()
        accu_attn_scores_bf16 = copy.deepcopy(accu_attn_scores)

        np.set_printoptions(threshold=10000000, suppress=True, precision=10)
        counter = 0

        context_bf16 = copy.deepcopy(context)
        '''
        for i, it in enumerate(context_bf16):
            if it is not None:
                if it.dtype == torch.float32:
                    context_bf16[i] = it.bfloat16()
        '''
        words_bf16 = copy.deepcopy(words)

        for idx in range(1, self.max_seq_len):
            print("***********************************************************************************************")
            if not len(active):
                assert not len(active_bf16)
                break
            counter += 1

            eos_mask = (words == EOS)
            eos_mask = eos_mask.view(-1, beam_size)

            eos_mask_bf16 = (words_bf16 == EOS)
            eos_mask_bf16 = eos_mask_bf16.view(-1, beam_size)
            print("")
            print(" eos_mask")
            print(torch.sum(eos_mask_bf16.eq(eos_mask).int() - 1))

            terminating, _ = eos_mask.min(dim=1)
            terminating_bf16, _ = eos_mask_bf16.min(dim=1)
            print("")
            print(" terminating")
            print(torch.sum(terminating.eq(terminating_bf16).int() - 1))

            lengths[active[~eos_mask.view(-1)]] += 1
            lengths_bf16[active_bf16[~eos_mask_bf16.view(-1)]] += 1
            print("")
            print(" lengths")
            print((lengths.abs().min(), lengths.abs().max(), lengths.min(), lengths.max(), (lengths_bf16 - lengths).abs().max()))

            self.model.debug_bf16_switch = False
            self.model.encoder.debug_bf16_switch = False
            self.model.decoder.debug_bf16_switch = False
            words, logprobs, attn, context, logits, logprobs_org = self.model.generate(words, context, beam_size)

            self.model.debug_bf16_switch = True
            self.model.encoder.debug_bf16_switch = False
            self.model.decoder.debug_bf16_switch = False
            words_bf16, logprobs_bf16, attn_bf16, context_bf16, logits_bf16, logprobs_org_bf16  = self.model.generate(words_bf16, context_bf16, beam_size)
            print("")
            print(" generate::logits")
            print((logits.abs().min(), logits.abs().max(), logits.min(), logits.max(), (logits_bf16 - logits).abs().max()))
            print("")
            print(" generate::logprobs_org")
            print((logprobs_org.abs().min(), logprobs_org.abs().max(), logprobs_org.min(), logprobs_org.max(), (logprobs_org_bf16 - logprobs_org).abs().max()))
            print("")
            print(" generate::words_bf16")

            res = words_bf16.eq(words)
            print(res)
            print(torch.sum(words_bf16.eq(words).int() - 1))
            print("")
            print(" attn cosym:")
            print((attn.abs().min(), attn.abs().max(), attn.min(), attn.max(), (attn_bf16 - attn).abs().max()))
            print(" context cosym:")
            for i, it in enumerate(context):
                context_i = context[i]
                context_bf16_i = context_bf16[i]
                print((context_i.abs().min(), context_i.abs().max(), context_i.min(), context_i.max(), (context_bf16_i - context_i).abs().max()))
            print(" logprobs cosym:")
            print((logprobs.abs().min(), logprobs.abs().max(), logprobs.min(), logprobs.max(), (logprobs_bf16 - logprobs).abs().max()))

            self.model.encoder.debug_bf16_switch = False
            self.model.decoder.debug_bf16_switch = False

            attn = attn.float().squeeze(attn_query_dim)
            attn = attn.masked_fill(eos_mask.view(-1).unsqueeze(1), 0)
            accu_attn_scores[active] += attn

            attn_bf16 = attn_bf16.float().squeeze(attn_query_dim)
            attn_bf16 = attn_bf16.masked_fill(eos_mask_bf16.view(-1).unsqueeze(1), 0)
            accu_attn_scores_bf16[active_bf16] += attn_bf16

            print("")
            print(" accu_attn_scores_bf16")
            print((accu_attn_scores.abs().min(), accu_attn_scores.abs().max(), accu_attn_scores.min(), accu_attn_scores.max(), (accu_attn_scores_bf16 - accu_attn_scores).abs().max()))

            # words: (batch, beam, k)
            words = words.view(-1, beam_size, beam_size)
            words = words.masked_fill(eos_mask.unsqueeze(2), EOS)

            words_bf16 = words_bf16.view(-1, beam_size, beam_size)
            words_bf16 = words_bf16.masked_fill(eos_mask_bf16.unsqueeze(2), EOS)
            print("")
            print(" words_bf16")
            print((words.abs().min(), words.abs().max(), words.min(), words.max(), torch.sum(words_bf16.eq(words).int() - 1)))

            # logprobs: (batch, beam, k)
            logprobs = logprobs.float().view(-1, beam_size, beam_size)
            logprobs_bf16 = logprobs_bf16.float().view(-1, beam_size, beam_size)
            print("")
            print(" logprobs1")
            print((logprobs.abs().min(), logprobs.abs().max(), logprobs.min(), logprobs.max(), (logprobs_bf16 - logprobs).abs().max()))

            if eos_mask.any():
                logprobs[eos_mask] = eos_beam_fill
                assert eos_mask_bf16.any()
                logprobs_bf16[eos_mask_bf16] = eos_beam_fill

            print("")
            print(" logprobs2")
            print((logprobs.abs().min(), logprobs.abs().max(), logprobs.min(), logprobs.max(), (logprobs_bf16 - logprobs).abs().max()))

            active_scores = scores[active].view(-1, beam_size)
            active_scores_bf16 = scores_bf16[active_bf16].view(-1, beam_size)
            print("")
            print(" active_scores")
            print((active_scores.abs().min(), active_scores.abs().max(), active_scores.min(), active_scores.max(), (active_scores_bf16 - active_scores).abs().max()))

            # new_scores: (batch, beam, k)
            new_scores = active_scores.unsqueeze(2) + logprobs
            new_scores_bf16 = active_scores_bf16.unsqueeze(2) + logprobs_bf16

            if idx == 1:
                new_scores[:, 1:, :].fill_(float('-inf'))
                new_scores_bf16[:, 1:, :].fill_(float('-inf'))

            new_scores = new_scores.view(-1, beam_size * beam_size)
            new_scores_bf16 = new_scores_bf16.view(-1, beam_size * beam_size)
            print("")
            print(" new_scores")
            print((new_scores.abs().min(), new_scores.abs().max(), new_scores.min(), new_scores.max(), (new_scores_bf16 - new_scores).abs().max()))

            # index: (batch, beam)
            _, index = new_scores.topk(beam_size, dim=1)
            _, index_bf16 = new_scores_bf16.topk(beam_size, dim=1)
            print("")
            print(" index")
            print((index.abs().min(), index.abs().max(), index.min(), index.max(), (index_bf16 - index).abs().max()))

            source_beam = index / beam_size
            source_beam_bf16 = index_bf16 / beam_size
            print("")
            print(" source_beam")
            print((source_beam.abs().min(), source_beam.abs().max(), source_beam.min(), source_beam.max(), (source_beam_bf16 - source_beam).abs().max()))

            new_scores = new_scores.view(-1, beam_size * beam_size)
            new_scores_bf16 = new_scores_bf16.view(-1, beam_size * beam_size)

            best_scores = torch.gather(new_scores, 1, index)
            best_scores_bf16 = torch.gather(new_scores_bf16, 1, index_bf16)
            print("")
            print(" best_scores")
            print((best_scores.abs().min(), best_scores.abs().max(), best_scores.min(), best_scores.max(), (best_scores_bf16 - best_scores).abs().max()))

            scores[active] = best_scores.view(-1)
            scores_bf16[active_bf16] = best_scores_bf16.view(-1)
            print("")
            print(" scores")
            print((scores.abs().min(), scores.abs().max(), scores.min(), scores.max(), (scores_bf16 - scores).abs().max()))

            words = words.view(-1, beam_size * beam_size)
            words = torch.gather(words, 1, index)
            words_bf16 = words_bf16.view(-1, beam_size * beam_size)
            words_bf16 = torch.gather(words_bf16, 1, index_bf16)

            # words: (1, batch * beam)
            words = words.view(word_view)
            words_bf16 = words_bf16.view(word_view)
            print("")
            print(" words")
            print((words.abs().min(), words.abs().max(), words.min(), words.max(), torch.sum(words_bf16.eq(words).int() - 1)))

            offset = global_offset[:source_beam.shape[0]]
            source_beam += offset.unsqueeze(1)

            offset_bf16 = global_offset[:source_beam_bf16.shape[0]]
            source_beam_bf16 += offset_bf16.unsqueeze(1)

            translation[active, :] = translation[active[source_beam.view(-1)], :]
            translation[active, idx] = words.view(-1)

            translation_bf16[active_bf16, :] = translation_bf16[active_bf16[source_beam_bf16.view(-1)], :]
            translation_bf16[active_bf16, idx] = words_bf16.view(-1)

            lengths[active] = lengths[active[source_beam.view(-1)]]
            lengths_bf16[active_bf16] = lengths_bf16[active_bf16[source_beam_bf16.view(-1)]]

            context[2] = context[2].index_select(1, source_beam.view(-1))
            context_bf16[2] = context_bf16[2].index_select(1, source_beam_bf16.view(-1))

            if terminating.any():
                assert terminating_bf16.any()

                not_terminating = ~terminating
                not_terminating = not_terminating.unsqueeze(1)
                not_terminating = not_terminating.expand(-1, beam_size).contiguous()

                not_terminating_bf16 = ~terminating_bf16
                not_terminating_bf16 = not_terminating_bf16.unsqueeze(1)
                not_terminating_bf16 = not_terminating_bf16.expand(-1, beam_size).contiguous()

                normalization_mask = active.view(-1, beam_size)[terminating]
                normalization_mask_bf16 = active_bf16.view(-1, beam_size)[terminating_bf16]

                # length normalization
                norm = lengths[normalization_mask].float()
                norm = (norm_const + norm) / (norm_const + 1.0)
                norm = norm ** norm_factor

                norm_bf16 = lengths_bf16[normalization_mask_bf16].float()
                norm_bf16 = (norm_const + norm_bf16) / (norm_const + 1.0)
                norm_bf16 = norm_bf16 ** norm_factor

                scores[normalization_mask] /= norm
                scores_bf16[normalization_mask_bf16] /= norm_bf16

                # coverage penalty
                penalty = accu_attn_scores[normalization_mask]
                penalty = penalty.clamp(0, 1)
                penalty = penalty.log()
                penalty[penalty == float('-inf')] = 0
                penalty = penalty.sum(dim=-1)

                penalty_bf16 = accu_attn_scores_bf16[normalization_mask_bf16]
                penalty_bf16 = penalty_bf16.clamp(0, 1)
                penalty_bf16 = penalty_bf16.log()
                penalty_bf16[penalty_bf16 == float('-inf')] = 0
                penalty_bf16 = penalty_bf16.sum(dim=-1)

                scores[normalization_mask] += cov_penalty_factor * penalty
                scores_bf16[normalization_mask_bf16] += cov_penalty_factor * penalty_bf16

                mask = base_mask[:len(active)]
                mask = mask.masked_select(not_terminating.view(-1))

                mask_bf16 = base_mask[:len(active_bf16)]
                mask_bf16 = mask_bf16.masked_select(not_terminating_bf16.view(-1))

                words = words.index_select(ctx_batch_dim, mask)
                context[0] = context[0].index_select(ctx_batch_dim, mask)
                context[1] = context[1].index_select(0, mask)
                context[2] = context[2].index_select(1, mask)

                words_bf16 = words_bf16.index_select(ctx_batch_dim, mask_bf16)
                context_bf16[0] = context_bf16[0].index_select(ctx_batch_dim, mask_bf16)
                context_bf16[1] = context_bf16[1].index_select(0, mask_bf16)
                context_bf16[2] = context_bf16[2].index_select(1, mask_bf16)

                active = active.masked_select(not_terminating.view(-1))
                active_bf16 = active_bf16.masked_select(not_terminating_bf16.view(-1))

                '''
                scores_tmp = scores.view(batch_size, beam_size).max(dim=1)
                _, idx_tmp = scores_tmp.max(dim=1)
                scores_tmp_bf16 = scores_bf16.view(batch_size, beam_size).max(dim=1)
                _, idx_tmp_bf16 = scores_tmp_bf16.max(dim=1)
                print("")
                print(" idx")
                print((idx_tmp.abs().min(), idx_tmp.abs().max(), idx_tmp.min(), idx_tmp.max(), torch.sum(idx_tmp_bf16.eq(idx_tmp).int() - 1)))
                '''

        scores = scores.view(batch_size, beam_size)
        _, idx = scores.max(dim=1)

        scores_bf16 = scores_bf16.view(batch_size, beam_size)
        _, idx_bf16 = scores_bf16.max(dim=1)

        print("")
        print(" idx")
        print((idx_tmp.abs().min(), idx_tmp.abs().max(), idx_tmp.min(), idx_tmp.max(),
               torch.sum(idx_tmp_bf16.eq(idx_tmp).int() - 1)))

        translation = translation[idx + global_offset, :]
        lengths = lengths[idx + global_offset]

        return translation, lengths, counter, scores
