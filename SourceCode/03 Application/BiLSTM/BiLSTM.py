import torch
import torch.nn as nn
from torch.autograd import Variable


class bilstm_attn(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, weights, bidirectional, dropout,
                 use_cuda, attention_size, sequence_length):
        super(bilstm_attn, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        # self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        # self.lookup_table.weight.data.uniform_(-1., 1.)
        self.lookup_table.weight = nn.Parameter(weights, requires_grad=False)

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(hidden_size * self.layer_size * 3, output_size)
        self.dropout = nn.Dropout(dropout)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
        # init_lstm( self.lstm )

    # self.attn_fc_layer = nn.Linear()

    def init_weights(self):
        initrange = 0.5
        self.label.weight.data.uniform_(-initrange, initrange)
        self.label.bias.data.zero_()

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = self.dropout(input)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            current_batch_size = self.batch_size
        else:
            current_batch_size = batch_size

        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size, current_batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.layer_size, current_batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.layer_size, current_batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size, current_batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        lstm_output = self.dropout(lstm_output)
        max_p = self.max_pool(lstm_output.permute(1, 2, 0)).squeeze(2)
        avg_p = self.avg_pool(lstm_output.permute(1, 2, 0)).squeeze(2)
        attn_output = self.attention_net(lstm_output)
        x = torch.cat((attn_output, max_p, avg_p), dim=1)
        logits = self.label(x)
        # output = self.softmax(logits)
        return logits, attn_output
