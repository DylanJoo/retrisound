import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from options import ModelOptions, TrainOptions

## prepare kwargs
R_model_name_or_path='facebook/contriever'
G_model_name_or_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model_opt = ModelOptions(
        retriever_name_or_path=R_model_name_or_path,
        generator_name_or_path=G_model_name_or_path,
)
train_opt = TrainOptions()

## prepare bi-encoders
tokenizer = AutoTokenizer.from_pretrained(R_model_name_or_path)
from modeling.rmt import RMTEncoder
from modeling.rife import Contriever
model = Contriever.from_pretrained(R_model_name_or_path)
encoder = deepcopy(model)
ada_encoder = RMTEncoder(
        base_model=model, 
        num_mem_tokens=4,
        tokenizer=tokenizer,
        input_size=512,
        sum_loss=False
)
from inbatch import InBatchInteraction
bi_encoders = InBatchInteraction(
        model_opt, 
        q_encoder=ada_encoder,
        d_encoder=encoder,
        fixed_d_encoder=True
)

## prepare generator
config = AutoConfig.from_pretrained(G_model_name_or_path)
generator = AutoModelForCausalLM.from_pretrained(
        G_model_name_or_path,
        config=config,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
)
        # attn_implementation="flash_attention_2"

from rag import RerankAugmentedGeneration
model = RerankAugmentedGeneration(llm=generator, biencoders=bi_encoders)

for n, p in model.named_parameters():
    if p.requires_grad:
        print(n)

# input = tokenizer(['hello world', 'apple'], return_tensors='pt', padding=True)
# input_ids = [input['input_ids'], input['input_ids']]
# attention_mask = [input['attention_mask'], input['attention_mask']]
# out = q_encoder(input_ids=input_ids, attention_mask=attention_mask)
# # print(out[0]['last_hidden_state_0'][0, :4, :10])
# # print(out[0]['last_hidden_state_1'][0, :4, :10])
# print(out[1][0])
# print(out[1][1])
