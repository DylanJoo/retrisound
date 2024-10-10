# [Retriever]
## Config & tokenizer
from modeling import SparseEncoder
from modeling.biencoders import SparseAdaptiveEncoders
from modeling.biencoders.sparse import RegularizationHead
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="facebook/contriever")
    add_pooling_layer: Optional[bool] = field(default=False)
    tau: Optional[float] = field(default=1.0)

model_opt = ModelOptions()
tokenizer = AutoTokenizer.from_pretrained(model_opt.retriever_name_or_path)
modifier = RegularizationHead(
    model_opt,
    encoder=SparseEncoder(
        model_name_or_path=model_opt.retriever_name_or_path,
        output='MLM', agg='max', activation='relu'
    ).train()
)
ada_retriever = SparseAdaptiveEncoders(
    model_opt,
    encoder=SparseEncoder(model_name_or_path=model_opt.retriever_name_or_path).eval(),
    modifier=modifier
)

input = tokenizer(['apple', 'apple2', 'apple3'], return_tensors='pt', padding=True)
input2 = tokenizer(['banana', 'banana2', 'banana3'], return_tensors='pt', padding=True)
input3 = tokenizer(['watermelon', 'watermelon2', 'watermelon3'], return_tensors='pt', padding=True)
input4 = tokenizer(['aaa', 'bbb', 'ccc'], return_tensors='pt', padding=True)
q_tokens=[input['input_ids'], input2['input_ids'], input3['input_ids'], input4['input_ids']],
q_masks=[input['attention_mask'], input2['attention_mask'], input3['attention_mask'], input4['input_ids']],

d_input = tokenizer(
    ['apple', 'apple and banana are good', 'apple and banana and watermelon are good'], 
    return_tensors='pt', 
    padding=True
)

out = ada_retriever.forward(
    q_tokens=[input['input_ids'], input2['input_ids'], input3['input_ids'], input4['input_ids']],
    q_masks=[input['attention_mask'], input2['attention_mask'], input3['attention_mask'], input4['input_ids']],
    d_tokens=[d_input['input_ids'], d_input['input_ids']],
    d_masks=[d_input['attention_mask'], d_input['attention_mask']],
)
print(out)
