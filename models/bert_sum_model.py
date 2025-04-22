# modules/bertsum_model.py
import torch
import torch.nn as nn
from transformers import BertModel

class BertSumFull(nn.Module):
    def __init__(self):
        super(BertSumFull, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.doc_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.doc_transformer = nn.TransformerEncoder(self.doc_encoder, num_layers=2)

        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_out.last_hidden_state  

        trans_out = self.doc_transformer(embeddings.permute(1, 0, 2))
        trans_out = trans_out.permute(1, 0, 2) 

        cls_output = trans_out[:, 0, :]
        probs = torch.sigmoid(self.classifier(cls_output)).squeeze()

        return probs
