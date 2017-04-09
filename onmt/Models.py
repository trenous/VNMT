import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
import math
import numpy
from tensorboard_logger import log_value



class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        inputSize = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

    def forward(self, input, hidden=None):
        batch_size = input.size(0) # batch first for multi-gpu compatibility
        emb = self.word_lut(input).transpose(0, 1)
        if hidden is None:
            h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            c_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            hidden = (h_0, c_0)

        outputs, hidden_t = self.rnn(emb, hidden)
        return hidden_t, outputs


class EncoderLatent(nn.Module):

    def __init__(self, opt):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        inputSize = opt.latent_vec_size

        super(EncoderLatent, self).__init__()
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def forward(self, input, hidden=None):
        batch_size = input.size(0) # batch first for multi-gpu compatibility
        if hidden is None:
            h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Variable(input.data.new(*h_size).zero_(), requires_grad=False)
            c_0 = Variable(input.data.new(*h_size).zero_(), requires_grad=False)
            hidden = (h_0, c_0)
        outputs, hidden_t = self.rnn(input.transpose(0,1), hidden)
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            layer = nn.LSTMCell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_%d' % i)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_size = opt.rnn_size
        self.out_size = opt.rnn_size

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.copy_(pretrained)


    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input).transpose(0, 1)

        batch_size = input.size(0)

        h_size = (batch_size, self.hidden_size)
        output = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for i, emb_t in enumerate(emb.chunk(emb.size(0), dim=0)):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs.transpose(0, 1), hidden, attn

class FeedForward(nn.Module):
    ''' FeedForward Module with one hidden layer
    '''
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        h_dim = (in_dim+out_dim) // 2
        self.linear_in = nn.Linear(in_dim, h_dim)
        self.activation = nn.Tanh()
        self.linear_out = nn.Linear(h_dim, out_dim)

    def forward(self, input):
        """
        input: in_dim
        returns: out_dim
        """
        hidden = self.linear_in(input)
        hidden = self.activation(hidden)
        out = self.linear_out(hidden)
        out = self.activation(out)
        return out

class LengthNet(nn.Module):
     ''' Computes parameters for the distribution
      of the latent length.
     In: Context Matrix batch x sourceL x dim
     Out: softmax(U * [softmax(wC + b1)C] + b2)
     '''
     def __init__(self, opt):
         super(LengthNet, self).__init__()
         self.attention = onmt.modules.ConvexCombination(opt)
         self.linear = FeedForward(opt.rnn_size, opt.max_len_latent)
         self.sm = nn.Softmax()

     def forward(self, context):
         '''
         in:
            context: batch x sourceL x dim
         out:
             pi:     batch x max_len_latent
         '''
         attn = self.attention(context)
         scores = self.linear(attn)
         pi = self.sm(scores)
         return pi



class GeneratorLatent(nn.Module):

    def __init__(self, opt):
        super(GeneratorLatent, self).__init__()
        self.activation = nn.Tanh()
        # Use same size as word embeddings
        self.size = opt.latent_vec_size
        self.sigma = nn.Linear(opt.rnn_size, self.size)
        self.mu = nn.Linear(opt.rnn_size, self.size)
        self.cuda = opt.cuda

    def forward(self, out):
        batch = out.size(0)
        eps =  Variable(torch.randn(batch, self.size),
                        requires_grad=False)
        if self.cuda:
            eps = eps.cuda()
        mu = self.mu(out)
        sigma = self.sigma(out)
        z_i = mu + torch.exp(sigma)*eps
        return z_i, mu, sigma

class DecoderLatent(nn.Module):

    def __init__(self, opt):
        super(DecoderLatent, self).__init__()
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        self.sample = opt.sample
        input_size = opt.latent_vec_size + 1 # Also Feed Length k at each step
        self.generator = GeneratorLatent(opt)
        input_size += opt.rnn_size # Also Feed Attention at each step
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttentionLatent(opt)
        self.dropout = nn.Dropout(opt.dropout)
        self.hidden_size = opt.rnn_size
        self.out_size = opt.latent_vec_size
        self.cuda = opt.cuda



    def forward(self, hidden, context, z_0, k):
        '''
        Samples a latent vector z from q(z|x,k)
        hidden: batch x num_layers x h_size
        context: batch x sourceL x rnn_size
        z_0: batch x latent_vec_size
        k: batch x 1
        '''
        h_0, c_0 = hidden
        k_max = int(torch.max(k.data)) #Longest Word in Batch to Sample
        batch_size = h_0.size(1)
        h_size = (batch_size, self.hidden_size)
        attn = self.attn(c_0[-1], context.t())
        z = []
        mu = []
        sigma = []
        z_i = z_0

        for i in xrange(k_max):
            z_i = torch.cat([z_i, k], 1) # Append Length To Input
            z_i = torch.cat((z_i, attn), 1)
            ### Fixed Hidden as hidden <= mask*hidden + (1-mask)*old_hidden
            output, (h_1, c_1) = self.rnn(z_i, hidden)
            mask = i * Variable(torch.ones(batch_size, 1),
                            requires_grad=False)
            if self.cuda:
                mask = mask.cuda()
            mask = mask.ge(k).unsqueeze(0).expand_as(h_1).float()
            h_1 = mask * hidden[0] + (1-mask) * h_1
            c_1 = mask * hidden[1] + (1-mask) * c_1
            hidden = (h_1, c_1)
            z_i, mu_i, sigma_i = self.generator(output)
            z += [z_i]
            mu += [mu_i]
            sigma += [sigma_i]
            ## ATTN based on output or z_i?
            attn = self.attn(output, context.t())
        z = torch.stack(z)
        mu = torch.stack(mu)
        sigma = torch.stack(sigma)
        # mask samples to length k
        mask = Variable(torch.range(0, float(k_max)-1), requires_grad=False)
        if self.cuda:
            mask = mask.cuda()
        mask = mask.unsqueeze(1).expand(mask.size(0), batch_size)
        mask = mask.ge(k.t().expand_as(mask))
        mask = mask.unsqueeze(2)
        z.masked_fill_(mask.expand_as(z), 0)
        mu.masked_fill_(mask.expand_as(mu), 0)
        sigma.masked_fill_(mask.expand_as(sigma), 0)

        return z.transpose(0, 1), mu.transpose(0,1), sigma.transpose(0,1), hidden

class NMTModel(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 generator,
                 opt):

        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.generate = False

    def set_generate(self, enabled):
        self.generate = enabled

    def make_init_decoder_output(self, context, dec):
        batch_size = context.size(1)
        h_size = (batch_size, dec.out_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        tgt = input[1][:, :-1]  # exclude last target from inputs
        ### Source Encoding
        enc_hidden, context = self.encoder(src)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        ### Target Decoding
        init_output = self.make_init_decoder_output(context,
                                                    self.decoder)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        out, dec_hidden, _attn = self.decoder(tgt,
                                              enc_hidden,
                                              context,
                                              init_output)

        if self.generate:
            out = self.generator(out)

        return out

class BaseLine(nn.Module):
    '''Input-dependent baseline for REINFORCE.
    '''

    def __init__(self, opt):
        super(BaseLine, self).__init__()
        self.convex = onmt.modules.ConvexCombination(opt)
        self.linear = FeedForward(opt.rnn_size, 1)

    def forward(self, contexts):
        return self.linear(self.convex(contexts.transpose(0,1)))


class Loss(nn.Module):
    '''Computes Variational Loss.
    '''

    def __init__(self, opt, generator, vocabSize):
        super(Loss, self).__init__()
        self.generator = generator

    def p_theta_y(self, output, targets):
        '''Computes Log Likelihood of Targets Given X.
        '''
        output = output.contiguous().view(-1, output.size(2))
        pred = self.generator(output)
        pred = pred.view(targets.size(0), targets.size(1), pred.size(1))
        gathered = torch.gather(pred, 2,  targets.unsqueeze(2)).squeeze()
        gathered = gathered.masked_fill_(targets.eq(onmt.Constants.PAD), 0)
        pty = torch.sum(gathered.squeeze(), 1)
        return pty



    def forward(self, outputs, targets, step=None):
        loss = -self.p_theta_y(outputs, targets)
        loss_report = loss.sum().data[0]
        loss = loss.mean()
        log_value('loss', loss.data[0], step)
        log_value('loss_report', loss_report, step)
        return loss, loss_report
