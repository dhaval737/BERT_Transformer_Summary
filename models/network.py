import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(n):
    return 0.5 * n * (1 + torch.tanh(math.sqrt(2 / math.pi) * (n + 0.044715 * torch.pow(n, 3))))


class PosFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        dimmodel(int): the size of input for the first-layer of the feedforwardnetwork.
        dimff(int): the hidden layer size of the 2nd-layer
            of the the feedforwardnetwork.
        dropout(float): dropout probability`.
    """

    def __init__(self, dimmodel, dimff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(dimmodel, dimff)
        self.w_2 = nn.Linear(dimff, dimmodel)
        self.layer_norm = nn.LayerNorm(dimmodel, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, n):
        internal = self.dropout_1(self.actv(self.w_1(self.layer_norm(n))))
        output = self.dropout_2(self.w_2(internal))
        return output + n


class MultiHeadedAttention(nn.Module):

    def __init__(self, headcount, dimmodel, dropout=0.1, uselinear=True):
        assert dimmodel % headcount == 0
        self.dimperhead = dimmodel // headcount
        self.dimmodel = dimmodel

        super().__init__()
        self.headcount = headcount

        self.linear_keys = nn.Linear(dimmodel, headcount * self.dimperhead)
        self.linear_values = nn.Linear(dimmodel, headcount * self.dimperhead)
        self.linear_query = nn.Linear(dimmodel, headcount * self.dimperhead)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.uselinear = uselinear
        if (self.uselinear):
            self.final_linear = nn.Linear(dimmodel, dimmodel)

    def forward(self, key, value, query, mask=None,
                layercache=None, type=None, predefinedgraph=None):
        """
        Computing the context vector and the attention vectors.
        """

        batchsize = key.size(0)
        dimperhead = self.dimperhead
        headcount = self.headcount
        keylen = key.size(1)
        querylen = query.size(1)

        def shape(n):
            """  projection """
            return n.view(batchsize, -1, headcount, dimperhead) \
                .transpose(1, 2)

        def unshape(n):
            """  compute context """
            return n.transpose(1, 2).contiguous() \
                .view(batchsize, -1, headcount * dimperhead)

        #Projecting key, value, and query.
        if layercache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                    self.linear_keys(query), \
                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layercache is not None:
                    device = key.device
                    if layercache["selfkeys"] is not None:
                        key = torch.cat(
                            (layercache["selfkeys"].to(device), key),
                            dim=2)
                    if layercache["selfvalues"] is not None:
                        value = torch.cat(
                            (layercache["selfvalues"].to(device), value),
                            dim=2)
                    layercache["selfkeys"] = key
                    layercache["selfvalues"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layercache is not None:
                    if layercache["memorykeys"] is None:
                        key, value = self.linear_keys(key), \
                            self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layercache["memorykeys"], \
                            layercache["memoryvalues"]
                    layercache["memorykeys"] = key
                    layercache["memoryvalues"] = value
                else:
                    key, value = self.linear_keys(key), \
                        self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        keylen = key.size(2)
        querylen = query.size(2)

        #Calculating and scale scores.
        query = query / math.sqrt(dimperhead)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask.byte(), -1e18)

        #Applying attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefinedgraph is None):
            attnmasked = attn[:, -1] * predefinedgraph
            attnmasked = attnmasked / \
                (torch.sum(attnmasked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attnmasked.unsqueeze(1)], 1)

        dropattn = self.dropout(attn)
        if (self.uselinear):
            context = unshape(torch.matmul(dropattn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(dropattn, value)
            return context