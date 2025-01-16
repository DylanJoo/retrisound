# [Retriever]
## Config & tokenizer
from modeling import SparseEncoder
from modeling.base_encoder_new import SparseEncoder
from modeling.biencoders.sparse_doc_crossattn import SparseAdaptiveEncoders
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="naver/splade-v3-doc")
    add_pooling_layer: Optional[bool] = field(default=False)
    tau: Optional[float] = field(default=1.0)

model_opt = ModelOptions()
tokenizer = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
encoder = SparseEncoder(model_opt.retriever_name_or_path, cross_attention=True)
ada_retriever = SparseAdaptiveEncoders(
    q_encoder=encoder,
    n_candidates=10
)

input1 = tokenizer(['apple', 'apple2', 'apple3'], return_tensors='pt', padding=True)
input2 = tokenizer(['banana', 'banana2', 'banana3'], return_tensors='pt', padding=True)
input3 = tokenizer(['watermelon', 'watermelon2', 'watermelon3'], return_tensors='pt', padding=True)
input4 = tokenizer(['aaa', 'bbb', 'ccc'], return_tensors='pt', padding=True)

q_tokens = [input1['input_ids'], input2['input_ids'], input3['input_ids'], input4['input_ids']]
q_masks = [input1['attention_mask'], input2['attention_mask'], input3['attention_mask'], input4['attention_mask']]

d_input = tokenizer(
    ['apple', 'apple and banana are good', 'apple and banana and watermelon are good'], 
    return_tensors='pt', 
    padding=True
)

q_out = ada_retriever.forward(q_tokens=q_tokens[0], q_masks=q_masks[0])

out = ada_retriever.forward(
    q_tokens=q_tokens[0], 
    q_masks=q_masks[0],
    f_tokens=q_tokens[1],
    f_masks=q_masks[1],
    d_tokens=[d_input['input_ids'], d_input['input_ids']],
    d_masks=[d_input['attention_mask'], d_input['attention_mask']],
    prev_output=q_out,
    step=1
)
print(out)
