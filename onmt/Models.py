import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
import math
import numpy
import ipdb
from tensorboard_logger import log_value
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


def make_mask(k):
    '''Returns mask for length vector k
       return: mask: batch_size x seq_length
    '''
    batch_size = k.size(0)
    k_max = torch.max(k.data)
    mask = Variable(torch.arange(0, k_max), requires_grad=False)
    if k.is_cuda:
        mask = mask.cuda()
    mask = mask.unsqueeze(0).expand(batch_size, mask.size(0))
    return mask.ge(k.expand_as(mask))

def mask_tensor(k, vec, fill):
    ''' Masks Entries in vec that are longer than specified by k.
        vec: batch_size x seq_length x ...
        k: batch_size x 1
    '''
    mask = make_mask(k)
    for i in range(2, vec.dim()):
        mask = mask.unsqueeze(i)
    vec.masked_fill_(mask.expand_as(vec), fill)
    return vec


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
            h_size = (self.layers * self.num_directions, batch_size,
                      self.hidden_size)
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

    def forward(self, input, k, hidden=None):
        batch_size = input.size(0) # batch first for multi-gpu compatibility
        if hidden is None:
            h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Variable(input.data.new(*h_size).zero_(), requires_grad=False)
            c_0 = Variable(input.data.new(*h_size).zero_(), requires_grad=False)
            hidden = (h_0, c_0)

        # FIXED: packed sequence for Samples with different length
        # Reorder inputs by Length
        k_sorted, indices = torch.sort(k.squeeze(), descending=True)
        lengths = k_sorted.data.tolist()
        input_sorted = torch.index_select(input, 0, indices).transpose(0,1)
        input_packed = pack(input_sorted, lengths)
        outputs_packed, hidden_t = self.rnn(input_packed, hidden)
        outputs_sorted, _ = unpack(outputs_packed)
        # Restore original order
        _, indices_reverse = torch.sort(indices, descending=False)
        outputs = torch.index_select(outputs_sorted, 1, indices_reverse)
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


    def forward(self, input, hidden, context, k, init_output):
        emb = self.word_lut(input).transpose(0, 1)

        batch_size = input.size(0)

        h_size = (batch_size, self.hidden_size)
        output = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        self.attn.applyMask(make_mask(k))
        for i, emb_t in enumerate(emb.chunk(emb.size(0), dim=0)):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(0,1))
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs.transpose(0, 1), hidden, attn

class FeedForward(nn.Module):
    ''' Simple FeedForward Module with one hidden layer
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

     def forward(self, context, mask_in):
         '''
         in:
            context: batch x sourceL x dim
         out:
             pi:     batch x max_len_latent
         '''

         self.attention.applyMask(mask_in)
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



    def forward(self, hidden, context, z_0, k, mask_in):
        '''
        Samples a latent vector z from q(z|x,k)
        hidden: batch x num_layers x h_size
        context: batch x sourceL x rnn_size
        z_0: batch x latent_vec_size
        k: batch x 1
        mask_in: batch x sourceL ByteTensor
        '''
        h_0, c_0 = hidden
        k_max = int(torch.max(k.data))  #Longest Word in Batch to Sample
        batch_size = h_0.size(1)
        h_size = (batch_size, self.hidden_size)
        self.attn.applyMask(mask_in)
        attn = self.attn(c_0[-1], context.transpose(0,1))
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
            attn = self.attn(output, context.transpose(0, 1))
        z = torch.stack(z).transpose(0,1)
        mu = torch.stack(mu).transpose(0,1)
        sigma = torch.stack(sigma).transpose(0,1)
        # mask samples to length k
        z = mask_tensor(k, z, 0)
        mu = mask_tensor(k, mu, 0)
        sigma = mask_tensor(k, sigma, 0)
        return z, mu, sigma, hidden

class NMTModel(nn.Module):

    def __init__(self,
                 encoder,
                 lengthnet,
                 decoderlatent,
                 encoderlatent,
                 decoder,
                 generator,
                 opt):

        super(NMTModel, self).__init__()
        self.sample = opt.sample
        self.sample_reinforce = opt.sample_reinforce
        self.lengthnet = lengthnet
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
        ''' Runs a Forward pass Through the model.
            Input: Minibatch of Sentences
            Returns:
              out:   List of decoder output states for each sample
              mu:    List of means of latent sequence for each sample drawn
              sigma: List of log-variances of latent sequence for each sample drawn
              z:     List of samples from standardnormal distribution
              pi:    Approximate Posterior Length Distribution
              k:     List of samples from pi
              context:
              Detached Context Vectors of Encoder for computation of Baseline
        '''
        src = input[0]
        mask_in = src.eq(onmt.Constants.PAD)
        tgt = input[1][:, :-1]  # exclude last target from inputs
        ### Source Encoding
        enc_hidden, enc_context = self.encoder(src)
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        ### Length
        pi = self.lengthnet(enc_context.transpose(0, 1), mask_in)
        k = []
        mu = []
        sigma = []
        out = []
        z = []
        ## Sample i times from length dist
        for i in range(self.sample):
            # Sample in [1, max_len]
            k_i = pi.multinomial().float() + 1.0
            k_i = Variable(k_i.data, requires_grad=False)
            k += [k_i]
            z_0 = self.make_init_decoder_output(enc_context, self.decoder_l)
            z_i, mu_i, sigma_i, dec_hidden_l = self.decoder_l(enc_hidden,
                                                              enc_context,
                                                              z_0, k_i, mask_in)
            ### Latent Encoding
            enc_hidden_l, context_l = self.encoder_l(z_i, k_i)
            ### Target Decoding
            init_output = self.make_init_decoder_output(context_l,
                                                        self.decoder)
            enc_hidden_l = (self._fix_enc_hidden(enc_hidden_l[0]),
                            self._fix_enc_hidden(enc_hidden_l[1]))
            out_i, dec_hidden, _attn = self.decoder(tgt, enc_hidden_l,
                                                    context_l, k_i,
                                                    init_output)
            mu += [mu_i]
            sigma += [sigma_i]
            z += [z_i]
            out += [out_i]

        return out, mu, sigma, pi, k, z, enc_context.clone().detach()

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
        self.gamma = opt.gamma
        self.max_len = opt.max_len_latent
        self.gpu = opt.cuda
        self.lam = opt.lam
        self.sample = opt.sample
        self.generator = generator
        ### NMTCriterion
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        self.crit = crit
        ### Latent Length Dist
        self.prior_len = torch.ones(self.max_len)
        for i in range(self.max_len):
            self.prior_len[i:] *= self.gamma
        self.prior_len /= self.prior_len.sum()
        self.prior_len = Variable(self.prior_len, requires_grad=False)
        self.prior_len = self.prior_len.unsqueeze(0)
        if self.gpu:
            self.prior_len = self.prior_len.cuda()
            self.crit = self.crit.cuda()
        self.r_mean = 0.0

    def kld_length(self, pi):
        ''' Returns the KL Divergence of the approximate posterior
           length distribution given and the prior.
        '''
        return (pi * torch.log(pi / self.prior_len.expand_as(pi))).mean(0).sum()

    def p_theta_y(self, output, targets):
        '''Computes Log Likelihood of Targets Given Z.
        '''
        scores = self.generator(output.contiguous().view(-1, output.size(2)))
        scores = scores.view(output.size(0), output.size(1), -1)
        pred = scores.max(2)[1]
        correct = pred.eq(targets).masked_select(targets.ne(onmt.Constants.PAD))
        logli = torch.gather(scores, 2,  targets.unsqueeze(2)).squeeze()
        logli = logli.masked_fill_(targets.eq(onmt.Constants.PAD), 0)
        pty = torch.sum(logli, 1)

        return pty, float(correct.sum().data[0])

    def forward(self, outputs, mu, sigma, pi, k , z, targets, kl_weight=1, baseline=None, step=None):
        batch_size = pi.size(0)
        kld_len = self.kld_length(pi)
        loss = kl_weight * kld_len.div(batch_size)
        loss_report = kl_weight * kld_len
        pty = 0.0
        kld = 0.0
        num_correct = 0.0
        rs = []
        kls = []
        ### Loss
        for i in range(self.sample):
            k_i = k[i]
            ### Compute KLD of Latent Sequence
            kld_ = 0.5*(torch.exp(2*sigma[i]) + mu[i]**2) - sigma[i] - 0.5
            kld_ = mask_tensor(k_i, kld_, 0)
            kld_ = kld_.view(kld_.size(0), kld_.size(1) * kld_.size(2))
            kld_  =  torch.sum(kld_, 1)
            kld += kld_.mean().div(self.sample)
            pty_, corr_ = self.p_theta_y(outputs[i], targets)
            pty += pty_.mean().div(self.sample)
            num_correct += corr_  / self.sample
            r_i = (pty_ - kl_weight*kld_)
            loss -= r_i.mean().div(self.sample)
            loss_report -= r_i.sum().clone().detach().div(self.sample)
            rs.append(r_i)
            kls.append(kld_)

        elbo = -loss.clone().detach().data[0]

        ### Return if in Eval Mode
        if not self.training:
            return elbo, loss_report.data[0]

        ### Reinforcements
        bl = baseline.clone().detach()
        rein_loss = 0.0
        for i in range(self.sample):
            indices = k[i] - 1
            # Copy Prevents Backprop Through Rewards
            r = rs[i].clone().detach()
            RE_grad = torch.log(torch.gather(pi, 1, indices.long()))
            reward_adjusted = r - self.r_mean - bl
            reinforcement = self.lam * reward_adjusted * RE_grad
            loss -= reinforcement.mean().div(self.sample)
        loss += rein_loss
        ### Baseline Loss
        r_avg = torch.stack(rs).mean(0).clone().detach()

        loss_bl = torch.pow(r_avg - baseline - self.r_mean, 2).mean()

        ### Update Running Average of Rewards
        if self.r_mean:
            self.r_mean = 0.95*self.r_mean + 0.05 * r_avg.mean().data[0]
        else:
            self.r_mean = r_avg.mean().data[0]

        ### Logging
        if self.sample > 1:
            klvar = torch.var(torch.stack(kls))
            log_value('STD KL Divergence', torch.sqrt(klvar).data[0], step)
        range_ = Variable(torch.arange(1, self.max_len + 1)).unsqueeze(0)
        if self.gpu:
            range_ = range_.cuda()
        E_pi = (pi * range_.expand_as(pi)).sum(1).mean()
        mean_sig = mask_tensor(k[0], torch.exp(sigma[0]), 0)
        mean_sig = mean_sig / k[0].unsqueeze(1).expand_as(mean_sig)
        mean_sig = mean_sig.sum(1).mean()
        log_value('BaseLine', baseline.mean().data[0], step)
        log_value('Expected Length', E_pi.data[0], step)
        log_value('Loss', loss.data[0], step)
        log_value('KLD', kld.data[0] , step)
        log_value('KLD_LEN', kld_len.data[0], step)
        log_value('p_y_given_z', pty.data[0], step)
        log_value('r_mean_step', r_avg.mean().data[0], step)
        log_value('r_moving_avg', self.r_mean, step)
        log_value('loss BL', loss_bl.data[0], step)
        log_value('ELBO', elbo, step)
        log_value('kl_weight', kl_weight, step)
        log_value('mean_sigma', mean_sig.data[0], step)

        return loss, loss_bl, loss_report.data[0], num_correct
