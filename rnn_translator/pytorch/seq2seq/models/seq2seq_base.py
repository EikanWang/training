import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import copy


class Seq2Seq(nn.Module):
    def __init__(self, encoder=None, decoder=None, batch_first=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_first = batch_first
        self.debug_bf16_switch = False

    def encode(self, inputs, lengths):
        return self.encoder(inputs, lengths)

    def decode(self, inputs, context, inference=False):
        return self.decoder(inputs, context, inference)

    def generate(self, inputs, context, beam_size):
        logits, scores, new_context = self.decode(inputs, context, True)
        logits = logits.float()
        logprobs = log_softmax(logits, dim=-1)
        logprobs_ = copy.deepcopy(logprobs)
        if self.debug_bf16_switch:
            logprobs, words = logprobs.topk(beam_size, dim=-1)
        else:
            logprobs, words = logprobs.bfloat16().topk(beam_size, dim=-1)
        return words, logprobs.float(), scores, new_context, logits, logprobs_
