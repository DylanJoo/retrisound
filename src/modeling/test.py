import torch
from copy import deepcopy
from transformers import AutoTokenizer
from base_encoder import Contriever

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = Contriever.from_pretrained('bert-base-uncased')

#######
# test q encder
######
from rmt import RMTEncoder
q_encoder = RMTEncoder(
    base_model=model, 
    num_mem_tokens=4,
    tokenizer=tokenizer,
    input_size=512,
    sum_loss=False
)

# input = tokenizer(['hello world', 'apple'], return_tensors='pt', padding=True)
# input_ids = [input['input_ids'], input['input_ids'], input['input_ids']]
# attention_mask = [input['attention_mask'], input['attention_mask'], input['attention_mask']]
# out = q_encoder(input_ids, attention_mask)

# print(out.keys())
# print(out['last_hidden_state'].shape)
# print(out['last_hidden_state'][0].shape)
# print(out['last_hidden_state'][0][:4, :10])
# print(out['last_hidden_state'][1][:4, :10])

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal
@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="facebook/contriever")
    add_pooling_layer: Optional[bool] = field(default=False)
    num_mem_tokens: Optional[int] = field(default=1)
    num_budget: Optional[int] = field(default=5)
    tau: Optional[float] = field(default=1.0)

from biencoders import AdaptiveReranker
model_opt = ModelOptions()
d_encoder = deepcopy(model)
ada_reranker = AdaptiveReranker(
    model_opt,
    q_encoder=q_encoder,
    d_encoder=d_encoder,
)

input = tokenizer(['apple'], return_tensors='pt', padding=True)
input2 = tokenizer(['banana'], return_tensors='pt', padding=True)
input3 = tokenizer(['watermelon'], return_tensors='pt', padding=True)
input_ids = [input['input_ids'], input2['input_ids'], input3['input_ids']]
attention_mask = [input['attention_mask'], input2['attention_mask'], input3['attention_mask']]

input = tokenizer(['apple'], return_tensors='pt', padding=True)
input2 = tokenizer(['apple and banana are good'], return_tensors='pt', padding=True)
input3 = tokenizer(['apple and banana and watermelon are good'], return_tensors='pt', padding=True)
d_input_ids = [input['input_ids'], input2['input_ids'], input3['input_ids']]
d_attention_mask = [input['attention_mask'], input2['attention_mask'], input3['attention_mask']]

out = ada_reranker.forward(
    input_ids,
    attention_mask,
    d_input_ids,
    d_attention_mask,
)

print(out.loss)
print(out.logits)
print(out.probs)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v0.6')
# tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v0.6')
# import evaluate
# reward_model = evaluate.load('rouge')
#
# from reward import GenerativeRLWrapper
# reward_value_model = GenerativeRLWrapper(
#     generator=model,
#     tokenizer=tokenizer,
#     reward_model=reward_model
# )
#
# q, r, r_texts, test = reward_value_model._inference(['hello, I am', 'the reason i am here is'])
# # print(r_texts)
#
# p = reward_value_model.get_logprobs(q, r)
#
# print(len(test[0]), len(p[0]))
# for i in [0, 1]:
#     for a, b in zip(test[i], p[i]):
#         if a != '</s>':
#             print(a, b)
#
#
# print(r_texts)
# rw = reward_value_model.get_rewards(["app", "Thank you"], r_texts)
# print(rw)
