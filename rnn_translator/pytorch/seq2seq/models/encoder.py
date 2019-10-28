import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import seq2seq.data.config as config


class ResidualRecurrentEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, num_layers=8, bias=True,
                 dropout=0, batch_first=False, math='fp32', embedder=None):

        super(ResidualRecurrentEncoder, self).__init__()
        self.batch_first = batch_first
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias,
                    batch_first=batch_first, bidirectional=True))

        self.rnn_layers.append(
            nn.LSTM((2 * hidden_size), hidden_size, num_layers=1, bias=bias,
                    batch_first=batch_first))

        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias,
                        batch_first=batch_first))

        self.dropout = nn.Dropout(p=dropout)

        self.math = math

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                        padding_idx=config.PAD)

    def forward(self, inputs, lengths):
        x = self.embedder(inputs)

        if self.math == 'bf16':
            x = x.bfloat16()

        # bidirectional layer
        x = pack_padded_sequence(x, lengths.cpu().numpy(),
                                 batch_first=self.batch_first)

        if self.math == 'bf16':
            assert x.data.dtype == torch.bfloat16
            x = x.to(torch.float32)

        x, _ = self.rnn_layers[0](x)

        if self.math == 'bf16':
            x = x.to(torch.bfloat16)

        x, _ = pad_packed_sequence(x, batch_first=self.batch_first)

        # 1st unidirectional layer
        x = self.dropout(x)

        if self.math == 'bf16':
            x = x.to(torch.float32)
        x, _ = self.rnn_layers[1](x)
        if self.math == 'bf16':
            x = x.to(torch.bfloat16)

        # the rest of unidirectional layers,
        # with residual connections starting from 3rd layer
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)

            if self.math == 'bf16':
                x = x.to(torch.float32)
            x, _ = self.rnn_layers[i](x)
            if self.math == 'bf16':
                x = x.to(torch.bfloat16)

            x = x + residual

        return x
