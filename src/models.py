import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerModel(nn.Module):
    def __init__(self, model_name, model_dir, dropout, n_fc, n_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=model_dir ) 

        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_fc, n_classes)

    def forward(self, left, right):
        l_pooled_output = self.backbone(
            left["input_ids"].cuda(),
            attention_mask=left["attention_mask"].cuda(),
            token_type_ids=left["token_type_ids"].cuda(),
            return_dict=False,
        )

        r_pooled_output = self.backbone(
            right["input_ids"].cuda(),
            attention_mask=right["attention_mask"].cuda(),
            token_type_ids=right["token_type_ids"].cuda(),
            return_dict=False,
        )

        embeds = torch.cat((l_pooled_output[0], r_pooled_output[0]), dim=-1)
        embeds = torch.mean(embeds, dim=1)
        
        embeds = self.drop(embeds)
        output = self.classifier(embeds)
        
        return output
