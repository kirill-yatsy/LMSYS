import torch
import torch.nn as nn
from transformers import AutoModel
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForSequenceClassification, AutoModelForSequenceClassification
from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)


# double_quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=False,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

double_quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False,
)
# double_quant_config = BitsAndBytesConfig(load_in_8bit=True , bnb_8bit_use_double_quant=True)


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        lora_config = LoraConfig(
            r=config.lora_rank,  # the dimension of the low-rank matrices
            lora_alpha=config.lora_alpha,  # scaling factor for LoRA activations vs pre-trained weight activations
            lora_dropout=config.drop_out,
            bias="none",
            inference_mode=False,
            task_type=TaskType.SEQ_CLS,
            target_modules=config.lora_modules,
        )

        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.model,
            # num_labels=config.num_classes,
            quantization_config=double_quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        self.backbone.config.pad_token_id = 0  # unk
        self.backbone.config.bos_token_id = 1
        self.backbone.config.eos_token_id = 2 

        self.backbone = get_peft_model(self.backbone, lora_config)
        # Trainable Parameters
        self.backbone.print_trainable_parameters()

        # get the hidden size of the transformer
        self.hidden_size = self.backbone.config.hidden_size

        self.head_layer = nn.Sequential(
            nn.Dropout(config.drop_out),
            nn.Linear(
                self.backbone.config.vocab_size, self.backbone.config.hidden_size
            ),
            nn.Tanh(),
            nn.Dropout(config.drop_out),
            nn.Linear(self.backbone.config.hidden_size, config.num_classes),
        )

    def forward(self, input):
        output = self.backbone(
            input["input_ids"].cuda(),
            attention_mask=input["attention_mask"].cuda()
        )

        # embeds = torch.cat((l_pooled_output[0], r_pooled_output[0]), dim=-1)
        output = output.logits.mean(dim=1)
        output = self.head_layer(output.type(torch.float32))

        return output
