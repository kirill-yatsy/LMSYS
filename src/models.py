import torch
import torch.nn as nn
from transformers import AutoModel
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# double_quant_config = BitsAndBytesConfig(load_in_8bit=True , bnb_8bit_use_double_quant=True)




class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        lora_config = LoraConfig(
            r=config.lora_rank,  # the dimension of the low-rank matrices
            lora_alpha = config.lora_alpha, # scaling factor for LoRA activations vs pre-trained weight activations
            lora_dropout=config.drop_out, 
            bias='none',
            inference_mode=False,
            task_type=TaskType.SEQ_CLS,
            target_modules=config.lora_modules
        )
        
        self.backbone = LlamaForSequenceClassification.from_pretrained(
            config.model,
            num_labels=config.num_classes, 
            quantization_config=double_quant_config, 
        )

        self.backbone.score = nn.Identity()

        self.backbone.config.pretraining_tp = 1 

        # Assign Padding TOKEN
        self.backbone.config.pad_token_id = config.pad_token_id

        self.backbone = get_peft_model(self.backbone, lora_config)
        # Trainable Parameters
        self.backbone.print_trainable_parameters()

        # get the hidden size of the transformer
        self.hidden_size = self.backbone.config.hidden_size

        self.head_layer = nn.Sequential(
            nn.Dropout(config.drop_out),
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.drop_out),
            nn.Linear(self.backbone.config.hidden_size, config.num_classes),
        )

    def forward(self, input):
        # l_pooled_output = self.backbone(
        #     left["input_ids"].cuda(),
        #     attention_mask=left["attention_mask"].cuda(),
        #     # token_type_ids=left["token_type_ids"].cuda(),
        #     return_dict=False,
        # )

         

        # r_pooled_output = self.backbone(
        #     right["input_ids"].cuda(),
        #     attention_mask=right["attention_mask"].cuda(),
        #     # token_type_ids=right["token_type_ids"].cuda(),
        #     return_dict=False,
        # )

        output = self.backbone(
            input["input_ids"].cuda(),
            attention_mask=input["attention_mask"].cuda(), 
            return_dict=False,
        )

        # embeds = torch.cat((l_pooled_output[0], r_pooled_output[0]), dim=-1)
         
        output = self.head_layer(output[0].type(torch.float32))
        
        return output
