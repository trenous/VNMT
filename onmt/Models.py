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



    def forward(self, hidden, context, z_0):
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

        for i in xrange(50):
            z_i = torch.cat([z_i, k], 1) # Append Length To Input
            z_i = torch.cat((z_i, attn), 1)
            ### Fixed Hidden as hidden <= mask*hidden + (1-mask)*old_hidden
            output, (h_1, c_1) = self.rnn(z_i, hidden)
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
        return z.transpose(0, 1), mu.transpose(0,1), sigma.transpose(0,1), hidden

class NMTModel(nn.Module):

    def __init__(self,
                 encoder,
                 decoderlatent,
                 encoderlatent,
                 decoder,
                 generator,
                 opt):

        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_l = decoderlatent
        self.encoder_l = encoderlatent
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

        ### Sample  from sequence given length
        z_0 = self.make_init_decoder_output(context, self.decoder_l)
        z, mu, sigma, hidden_l = self.decoder_l(enc_hidden,
                                                context, z_0)
        ### Latent Encoding
        enc_hidden, context = self.encoder_l(z)
        ### Target Decoding
        init_output = self.make_init_decoder_output(context,
                                                    self.decoder)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        out, dec_hidden, _attn = self.decoder(tgt,
                                              enc_hidden,
                                              context,
                                              init_output)
        z_p = Variable(torch.randn(z.size()), requires_grad=False).cuda()
        
        enc_hidden_p, context_p = self.encoder_l(z_p)
        init_output = self.make_init_decoder_output(context_p,
                                                    self.decoder)
        enc_hidden_p = (self._fix_enc_hidden(enc_hidden_p[0]),
                        self._fix_enc_hidden(enc_hidden_p[1]))
        out_p, dec_hidden_p, _attn_p = self.decoder(tgt,
                                                    enc_hidden_p,
                                                    context_p,
                                                    init_output)
                
        if self.generate:
            out = self.generator(out)

        return out, mu, sigma, pi, k, z, out_p


class Loss(nn.Module):
    '''Computes Variational Loss.
    '''

    def __init__(self, opt, generator, vocabSize):
        super(Loss, self).__init__()
        self.generator = generator

    def p_theta_z(self, z):
        '''Returns Log Density of z given length k under the Prior.'''
        log_pz = -0.5*z*z - math.log(math.sqrt(2*math.pi))
        return torch.sum(log_pz)


    def q_phi(self, mu, sigma, z):
        '''Returns Log Density of z given length k under the
           approximate posterior q_phi(z | x, k).
        '''
	log_qf = -0.5 * torch.pow((z-mu), 2)
        log_qf = log_qf / torch.pow(torch.exp(sigma), 2)
        log_qf -= sigma + math.log(math.sqrt(2*math.pi))
        return torch.sum(log_qf)

    def p_theta_y(self, output,  targets):
        '''Computes Log Likelihood of Targets Given X.
        '''
        output = output.contiguous().view(-1, output.size(2))
        pred = self.generator(output)
        pred = pred.view(targets.size(0), targets.size(1), pred.size(1))
        gathered = torch.gather(pred, 2,  targets.unsqueeze(2)).squeeze()
        gathered = gathered.masked_fill_(targets.eq(onmt.Constants.PAD), 0)
        pty = torch.sum(gathered.squeeze(), 1)
        return pty



    def forward(self, outputs, out_p, mu, sigma, pi , z, targets, kl_weight=1, step=None):
        batch_size = z.size(0)
        print 'batch_size: ', batch_size
        loss = 0.
        loss_report = 0.
        ### Compute log p_theta(y|z_ij):
        pty = self.p_theta_y(outputs,  targets)
        pty_p = self.p_theta_y(out_p,  targets)
        ### Compute log p_theta(z_ij)
        ptz = self.p_theta_z(z)
        ### Compute log q_phi(z_ij | x)
        qfz = self.q_phi(mu, sigma, z) 
        loss -= pty - kl_weight *(qfz + ptz)
        loss_report = loss.data[0]
        loss = loss.div(batch_size)
        if not self.training:
            return None,  loss_report
        log_value('KLD', ( (qfz - ptz).div(batch_size)).data[0], step)
        log_value('p_y_given_z', pty.div(batch_size).data[0], step)
        log_value('p_y', pty_p.div(batch_size).data[0], step)
        log_value('loss', loss.data[0], step)
        log_value('loss_report', loss_report, step)
        return loss, loss_report
