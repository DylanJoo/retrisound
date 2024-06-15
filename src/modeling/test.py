import torch
from rmt import RMTEncoder
from rife import Contriever
from copy import deepcopy

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = Contriever.from_pretrained('bert-base-uncased')
d_encoder = deepcopy(model)
q_encoder = RMTEncoder(
    base_model=model, 
    num_mem_tokens=4,
    tokenizer=tokenizer,
    input_size=512,
    sum_loss=False
)

input = tokenizer(['hello world', 'apple'], return_tensors='pt', padding=True)
input_ids = [input['input_ids'], input['input_ids']]
attention_mask = [input['attention_mask'], input['attention_mask']]
out = q_encoder(input_ids=input_ids, attention_mask=attention_mask)
# print(out[0]['last_hidden_state_0'][0, :4, :10])
# print(out[0]['last_hidden_state_1'][0, :4, :10])
print(out[1][0])
print(out[1][1])

