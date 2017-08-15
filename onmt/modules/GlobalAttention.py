"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a

Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:

"""
import ipdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        attn = attn - attn.max(1)[0].expand_as(attn)
        attn = torch.exp(attn)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, float(0))
        attn = attn / attn.sum(1).expand_as(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn

class GlobalAttentionLatent(nn.Module):
    def __init__(self, opt):
        dim = opt.rnn_size
        super(GlobalAttentionLatent, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """

        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        attn = attn - attn.max(1)[0].expand_as(attn)
	attn = torch.exp(attn)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask.data, float(0.))
        attn = attn / attn.sum(1).expand_as(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim

        return weightedContext

class ConvexCombination(nn.Module):
    ''' Computes Convex Combination of Context Vectors.
        In: Context Matrix C batch x sourceLength x dim
        Out: Convex Combination of the columns of C batch x dim
    '''
    def __init__(self, opt):
        dim = opt.rnn_size
        super(ConvexCombination, self).__init__()
        self.sm = nn.Softmax()
        self.linear_in = nn.Linear(dim, 1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, context):
        """
        context: batch x sourceL x dim
        """
        # Compute Scores
        batch = context.size(0)
        length = context.size(1)
        dim = context.size(2)
        contextv = context.contiguous().view(batch*length, dim)
        scores = self.linear_in(contextv).view(batch, length, 1).squeeze(2)
        # scores.size =  batch x sourceL
        # Compute Convex Combination
        score = scores - scores.max(1)[0].expand_as(scores)
	attn = torch.exp(scores)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask.data, float(0.))
        attn = attn / attn.sum(1).expand_as(attn)
        attn = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        weightedContext = torch.bmm(attn, context)  # batch x dim
        return weightedContext.squeeze(1)
