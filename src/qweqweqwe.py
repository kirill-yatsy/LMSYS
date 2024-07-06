from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from transformers import (
    AutoTokenizer,
    LlamaModel,
    LlamaForSequenceClassification,
    GenerationConfig,
)
from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)
import torch
import pandas as pd
import transformers
import re

tokenizer = AutoTokenizer.from_pretrained("WeOpenML/PandaLM-7B-v1", use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    "WeOpenML/PandaLM-7B-v1",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

 

tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

def build_prompt(instruction, input, resp1, resp2, result=None, explain=None, ref=None):
    rsp = f"### Response 1:\n{resp1}\n\n### Response 2:\n{resp2}"

    if input:
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n{rsp}\n\n### Evaluation:\n"
    else:
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:\n{instruction}\n\n{rsp}\n\n### Evaluation:\n"

    if result:
        output_sequence = f"{result}\n\n### Reason: {explain}\n\n### Reference: {ref}\n"
        return input_sequence, output_sequence
    else:
        return input_sequence
    

prompt = build_prompt(
    "Find response that user would most likely prefer.",
    "tell something interesting about Shanghai",
    "Here are some interesting facts about Shanghai: Shanghai is the largest city in China with a population of over 24 million people. It is a global financial hub and a transport hub with the world\'s busiest container port. • Shanghai has a very cosmopolitan feel to it. It was once called the \"Paris of the East\". You can see many European-style buildings and architecture in Shanghai, especially in the Bund area.\n\n• Shanghai Tower is the tallest building in Shanghai and the second tallest in the world at 632 meters. The top floor observatory offers stunning views of the city skyline.\n\n• Shanghai has a rich cultural life. It is home to many museums, theaters, and performance venues. Acclaimed Shanghai Symphony Orchestra regularly performs at the Shanghai Oriental Art Center.\n\n• Shanghai cuisine is world famous. Some signature dishes like Xiaolongbao (soup dumplings), Szechuan cuisine, and braised pork belly are popular among locals and tourists alike.\n\n• Shanghai Disneyland Park is the first Disney theme park resort in mainland China. It features many exciting rides and attractions as well as spectacular parades and fireworks.\n\n• The French Concession is a popular historical area in Shanghai with tree-lined avenues, European-style villas, and charming cafes. It offers a glimpse into the city\'s colonial past.\n\n• Shanghai has a fast and convenient public transit system, including buses, metros, ferries and high-speed rail. It makes getting around this large city very easy for tourists.\n\n• Shanghai experiences a humid subtropical climate with hot, humid summers and chilly, damp winters. The best times to visit are spring (March to May) and autumn (September to October).\n\nThose are some interesting facts about the amazing city of Shanghai. Let me know if you would like to know more.",
    "Shanghai is a city in China that is known for its fast-paced lifestyle, its many skyscrapers, and its delicious food. It is the largest city in China by population, with over 24 million people living in the city proper. Shanghai is also one of the most important financial centers in the world.\n\nOne of the most interesting things about Shanghai is its history. The city was founded in the 12th century, but it did not become a major city until the 19th century. In the 1840s, Shanghai was opened to foreign trade, and it quickly became a major center of commerce. The city was also a major center of the Chinese Revolution in the early 20th century.\n\nShanghai is a very diverse city, with people from all over China and the world living there. The city is also home to a large number of expatriates. This diversity is one of the things that makes Shanghai such a vibrant and exciting place to live.\n\nShanghai is a city that is constantly changing. The skyline is constantly changing as new skyscrapers are built. The city is also becoming more and more international, as more and more people from all over the world move to Shanghai.\n\nIf you are looking for a city that is full of life, culture, and opportunity, then Shanghai is the place for you. It is a city that is constantly changing and evolving, and it is sure to keep you entertained for years to come.\n\nHere are some other interesting facts about Shanghai:\n\n* Shanghai is the most populous city in China, with over 24 million people living in the city proper.\n* Shanghai is one of the most important financial centers in the world.\n* Shanghai is a very diverse city, with people from all over China and the world living there.\n* The skyline of Shanghai is constantly changing as new skyscrapers are built.\n* Shanghai is becoming more and more international, as more and more people from all over the world move to Shanghai.\n* Shanghai is a city that is full of life, culture, and opportunity.",
)

prepared_input = tokenizer(
    prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=1024
)

print(prompt)

model.eval()

input_ids = prepared_input["input_ids"].to("cuda")
generation_config = GenerationConfig(
    temperature=0,
    top_p=1,
    top_k=1,
    num_beams=4,
    early_stopping=True,
    repetition_penalty=1.2,
)

with torch.no_grad():
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=50,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

print(output)