import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
import math
import numpy
from tensorboard_logger import log_value
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


def make_mask(k):
    '''Returns mask for length vector k
       return: mask: batch_size x seq_length
    '''
    batch_size = k.size(0)
    k_max = torch.max(k.data)
    mask = torch.Tensor(torch.arange(0, k_max))
    if k.is_cuda:
        mask = mask.cuda()
    mask = mask.unsqueeze(0).expand(batch_size, mask.size(0))
    return mask.ge(k.data.expand_as(mask))

def mask_tensor(k, vec, fill):
    ''' Masks Entries in vec that are longer than specified by k.
        vec: batch_size x seq_length x ...
        k: batch_size x 1
        '''

    mask = make_mask(k)
    for i in range(2, vec.dim()):
        mask = mask.unsqueeze(i)
    vec.data.masked_fill_(mask.expand_as(vec), fill)
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

    def forward(self, input, k, hidden=None):
        batch_size = input.size(0) # batch first for multi-gpu compatibility
        if hidden is None:
            h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Variable(input.data.new(*h_size).zero_(), requires_grad=False)
            c_0 = Variable(input.data.new(*h_size).zero_(), requires_grad=False)
            hidden = (h_0, c_0)

        # FIXME: packed sequence
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
        '''
        h_0, c_0 = hidden
        k_max = int(torch.max(k.data))  #Longest Word in Batch to Sample
        batch_size = h_0.size(1)
        h_size = (batch_size, self.hidden_size)
        self.attn.applyMask(mask_in.data)
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
        z = torch.stack(z).transpose(0,1)
        mu = torch.stack(mu).transpose(0,1)
        sigma = torch.stack(sigma).transpose(0,1)
        # mask samples to length k
        z = mask_tensor(k, z, 0)
        mu = mask_tensor(k, mu, 0)
        sigma = mask_tensor(k, sigma, 1)
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
        src = input[0]
        mask_in = src.eq(onmt.Constants.PAD)
        tgt = input[1][:, :-1]  # exclude last target from inputs
        ### Source Encoding
        enc_hidden, context = self.encoder(src)
        ### Should Detach Contexts for Baseline???
        context_ = context.clone().detach()
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        ### Length
        pi = self.lengthnet(context.t(), mask_in)
        k = []
        mu = []
        sigma = []
        out = []
        z = []
        ## Sample i times from length dist
        for i in range(self.sample):
            mu += [[]]
            sigma += [[]]
            out += [[]]
            z += [[]]
            # Sample in [1, max_len]
            k_i = pi.multinomial().float() + 1.0
            k_i = Variable(k_i.data, requires_grad=False)
            k += [k_i]
            ### Sample j times from sequence given length
            for j in range(self.sample_reinforce):
                z_0 = self.make_init_decoder_output(context, self.decoder_l)
                z_ij, mu_ij, sigma_ij, hidden_l = self.decoder_l(enc_hidden,
                                                                 context, z_0, k_i, mask_in)
                ### Latent Encoding
                enc_hidden, context = self.encoder_l(z_ij, k_i)
                ### Target Decoding
                init_output = self.make_init_decoder_output(context,
                                                            self.decoder)
                enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                              self._fix_enc_hidden(enc_hidden[1]))
                out_ij, dec_hidden, _attn = self.decoder(tgt,
                                                         enc_hidden,
                                                         context, k_i,
                                                         init_output)
                mu[i] += [mu_ij]
                sigma[i] += [sigma_ij]
                z[i] += [z_ij]
                out[i] += [out_ij]
                if self.generate:
                    out = self.generator(out)
        return out, mu, sigma, pi, k, z, context_

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
        self.lam = opt.lam
        self.sample = opt.sample
        self.reinforce = opt.sample_reinforce
        ### NMTCriterion
        weight = torch.ones(vocabSize)
        weight[onmt.Constants.PAD] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        self.gpu = opt.cuda
        self.crit = crit
        ### Latent Length Dist
        self.gamma = opt.gamma
        self.max_len = opt.max_len_latent
        self.prior_len = torch.ones(self.max_len)
        for i in range(self.max_len):
            self.prior_len[i:] *= self.gamma
        self.prior_len /= self.prior_len.sum()
        self.prior_len = Variable(self.prior_len, requires_grad=False)
        self.prior_len = self.prior_len.unsqueeze(0)
        if self.gpu:
            self.prior_len = self.prior_len.cuda()
            self.crit = self.crit.cuda()
        self.r_mean = 0

    def kld_length(self, pi):
        ''' Returns the KL Divergence of the approximate posterior
           length distribution given by pi and the prior.
        '''
        return (pi * torch.log(pi / self.prior_len.expand_as(pi))).sum()

    def p_theta_y(self, output, targets):
        '''Computes Log Likelihood of Targets Given X.
        '''
        output = output.contiguous().view(-1, output.size(2))
        scores = self.generator(output)
        pred = scores.max(1)[1]
        num_correct = pred.data.eq(targets.data).masked_select(targets.ne(onmt.Constants.PAD).data).sum()
        pred = scores.view(targets.size(0), targets.size(1), scores.size(1))
        gathered = torch.gather(pred, 2,  targets.unsqueeze(2)).squeeze()
        gathered = gathered.masked_fill_(targets.eq(onmt.Constants.PAD), 0)
        pty = torch.sum(gathered.squeeze(), 1)
        return pty, num_correct

    def forward(self, outputs, mu, sigma, pi, k , z, targets, kl_weight=1, baseline=None, step=None):
        batch_size = pi.size(0)
        ### Track Expexted Value of Length
        if self.training:
            range_ = Variable(torch.arange(1, self.max_len + 1)).unsqueeze(0)
            if self.gpu:
                range_ = range_.cuda()
            E_pi = (pi * range_.expand_as(pi)).sum(1).mean()
            log_value('Expected Length', E_pi.data[0], step)
        loss = 0.
        loss_report = 0.
        rs = []
        pty_ = 0.
        kld = 0.
        num_correct = 0.0
        kls = []
        for i in range(self.sample):
            k_i = k[i]
            for j in range(self.reinforce):
                ### Compute log p_theta(y|z_ij):
                klz_ = 0.5*(torch.exp(2*sigma[i][j]) + mu[i][j]**2) - sigma[i][j]
                klz_ -= 0.5
                klz_ = mask_tensor(k_i, klz_, 0)
                klz_ = klz_.view(klz_.size(0), klz_.size(1) * klz_.size(2))
                klz  =  torch.sum(klz_, 1)
                kld += klz.mean().div(self.sample).data[0]
                pty, corr_ = self.p_theta_y(outputs[i][j], targets)
                pty_ += pty.mean()
                num_correct += corr_ * (1.0 / (self.sample * self.reinforce))
                ### Compute log p_theta(z_ij)
                kls.append(klz.mean())
                if not j:
                    r_i = (pty - kl_weight*(klz))
                else:
                    r_i += (pty - kl_weight*klz)
            loss -= r_i.mean()
            loss_report -= r_i.sum().clone().detach()
            rs.append(r_i)
        kls = torch.mean(torch.stack(kls), 0)
        sg = torch.exp(sigma[0][0]).mean()
        if self.sample > 1:
            klvar = torch.var(torch.stack(kls), 0).mean()
            log_value('STD KL Divergence', torch.sqrt(klvar).data[0], step)
        ### TODO: If self.reinforce > 1, Need to normalize (?)
        ### OR: Remove self.reinforce
        r_sum = torch.stack(rs).clone().detach()
	r_mean = torch.mean(r_sum)
        r_sum = torch.sum(r_sum, 0).squeeze(0)
        kld_len = self.kld_length(pi)
        elbo = -loss.clone().detach().div(self.sample) - kld_len.div(batch_size)
        if not self.training:
            return None, elbo.data[0], (loss_report.div(self.sample) + kld_len).data[0]
        if self.r_mean:
            self.r_mean = 0.9*self.r_mean + 0.1 * r_mean.data[0]
        else:
            self.r_mean = r_mean.data[0]
        loss_bl = torch.pow(r_sum.div(self.sample) - baseline - self.r_mean, 2).mean()
        bl = Variable(baseline.data, requires_grad=False)
        for i in range(self.sample):
            indices = k[i] - 1
            # Copy Prevents Backprop Through Rewards
            r = Variable(rs[i].data, requires_grad=False)
            RE_grad = torch.log(torch.gather(pi, 1, indices.long()))
            reward_adjusted = r - self.r_mean - bl
            reinforcement = self.lam * reward_adjusted * RE_grad
            loss -= reinforcement.mean()
        loss = loss.div(self.sample)
        loss_report = loss_report.div(self.sample)
        loss += kl_weight * kld_len.div(batch_size)
        loss_report += kld_len
        log_value('KLD', (kld_len.div(batch_size) + torch.stack(kls).mean()).data[0], step)
        log_value('KLZ', kld, step)
        log_value('KLD_LEN', kld_len.div(batch_size).data[0], step)
        log_value('p_y_given_z', pty_.div(self.sample).data[0], step)
        log_value('r_mean_step', r_mean.data[0], step)
        log_value('r_moving_avg', self.r_mean, step)
        log_value('loss', loss.data[0], step)
        log_value('loss BL', loss_bl.data[0], step)
        log_value('ELBO', elbo.data[0], step)
        log_value('kl_weight', kl_weight, step)
        log_value('mean_sigma', sg.data[0], step)
        return loss, loss_bl, loss_report.data[0], num_correct
