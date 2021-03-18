import math
import torch
import torch.nn as nn
from models.network import MultiHeadedAttention, PosFeedForward

class PosEncoding(nn.Module):
    def __init__(self, dropout, dim, maxlen=5000):
        pe = torch.zeros(maxlen, dim)
        position = torch.arange(0, maxlen).unsqueeze(1)
        divterm = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * divterm)
        pe[:, 1::2] = torch.cos(position.float() * divterm)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, embed, step=None):
        embed = embed * math.sqrt(self.dim)
        if step:
            embed = embed + self.pe[:, step][:, None, :]

        else:
            embed = embed + self.pe[:, : embed.size(1)]
        embed = self.dropout(embed)
        return embed

    def get_emb(self, embed):
        return self.pe[:, : embed.size(1)]


class ExTransformerEncoderLayer(nn.Module):
    def __init__(self, dimmodel, heads, dimff, dropout):
        
        super().__init__()

        self.self_attn = MultiHeadedAttention(heads, dimmodel, dropout=dropout)
        self.feed_forward = PosFeedForward(dimmodel, dimff, dropout)
        self.layer_norm = nn.LayerNorm(dimmodel, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dimmodel, dimff, heads, dropout, internallayers=0):
        super().__init__()
        self.dimmodel = dimmodel
        self.internallayers = internallayers
        self.pos_emb = PosEncoding(dropout, dimmodel)
        self.transformer_inter = nn.ModuleList(
            [ExTransformerEncoderLayer(dimmodel, heads, dimff, dropout) for i in range(internallayers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dimmodel, eps=1e-6)
        self.wo = nn.Linear(dimmodel, 1, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, top, mask):
        batchsize, numofsents = top.size(0), top.size(1)
        pos_emb = self.pos_emb.pe[:, :numofsents]
        n = top * mask[:, :, None].float()
        n = n + pos_emb

        for i in range(self.internallayers):
            n = self.transformer_inter[i](i, n, n, 1 - mask) 
        n = self.layer_norm(n)
        sent_scores = self.sigmoid(self.wo(n))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
