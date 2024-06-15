import inspect
from rmt import RMTEncoder
from rife import Contriever
from copy import deepcopy

from transformers import AutoTokenizer
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')
# d_encoder = deepcopy(model)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
q_encoder = RMTEncoder(
    base_model=model, 
    num_mem_tokens=4,
    tokenizer=tokenizer,
    input_size=512,
    sum_loss=False
)

input = tokenizer('hello world', return_tensors='pt', padding=True)
input_ids = [input['input_ids'], input['input_ids']]
attention_mask = [input['attention_mask'], input['attention_mask']]
# print(d_encoder(**input))
out = q_encoder(input_ids=input_ids, attention_mask=attention_mask)

