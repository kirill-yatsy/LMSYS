import torch
import torch.nn as nn
from transformers import AutoModel
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import BitsAndBytesConfig

# double_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
double_quant_config = BitsAndBytesConfig(load_in_8bit=True , bnb_8bit_use_double_quant=True)

class TransformerModel(nn.Module):
    def __init__(self, model_name, model_dir, dropout, n_fc, n_classes):
        super().__init__()
        self.backbone = DebertaV2PairRM.from_pretrained(model_name, cache_dir=model_dir ) 
        self.backbone = self.backbone.pretrained_model

        # get the hidden size of the transformer
        self.hidden_size = self.backbone.config.hidden_size
 
        self.head_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*self.hidden_size, 1*self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(1 * self.hidden_size, n_classes),
        )

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
        embeds = torch.mean(embeds, dim=1).type(torch.float32)
         
        output = self.head_layer(embeds)
        
        return output
