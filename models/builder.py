import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from models.encoder import TransformerEncoder

class Bert(nn.Module):
    def __init__(self, bert_type='bertbase'):
        super(Bert, self).__init__()
        self.bert_type = bert_type

        if bert_type == 'bert_base':
            config = BertConfig()
            self.model = BertModel(config)

    def forward(self, x, segments, mask):
        top, _ = self.model(x, attention_mask=mask, token_type_ids=segments)
        return top

class SummarizerLayer(nn.Module):
    def __init__(self, device, pretrained=None, bert_type='bert_base'):
        super().__init__()
        self.device = device
        self.bert = Bert(bert_type=bert_type)
        self.ext_layer = TransformerEncoder(
            self.bert.model.config.hidden_size, dimff=2048, heads=8, dropout=0.2, internallayers=2
        )

        if pretrained is not None:
            self.load_state_dict(pretrained, strict=True)
        self.to(device)

    def forward(self, source, segments, clss, srcmask, clsmask):
        top = self.bert(source, segments, srcmask)
        sents = top[torch.arange(top.size(0)).unsqueeze(1), clss]
        sents = sents * clsmask[:, :, None].float()
        sent_scores = self.ext_layer(sents, clsmask).squeeze(-1)
        return sent_scores, clsmask
