import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from base_encoder import Contriever

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = Contriever.from_pretrained('bert-base-uncased')

def test_q_encoder():
    from base_encoder import Contriever
    q_encoder = Contriever.from_pretrained('facebook/contriever-msmarco')

    input = tokenizer(['hello world', 'apple'], return_tensors='pt', padding=True)
    out = q_encoder(**input)
    return q_encoder

def test_bi_encder():
    from dataclasses import dataclass, field
    from typing import Optional, Union, Tuple, Literal
    @dataclass
    class ModelOptions:
        retriever_name_or_path: Optional[str] = field(default="facebook/contriever")
        add_pooling_layer: Optional[bool] = field(default=False)
        num_mem_tokens: Optional[int] = field(default=1)
        num_budget: Optional[int] = field(default=5)
        tau: Optional[float] = field(default=1.0)

    # from biencoders import AdaptiveReranker
    from modifier import FeedbackQueryModifier
    model_opt = ModelOptions()
    qr_encoder = test_q_encoder()
    qf_encoder = deepcopy(qr_encoder)

    biencoder = FeedbackQueryModifier(
        model_opt,
        qr_encoder=qr_encoder,
        qf_encoder=qf_encoder,
    )

    input = tokenizer(['apple'], return_tensors='pt', padding=True)
    input2 = tokenizer(['banana'], return_tensors='pt', padding=True)
    input3 = tokenizer(['watermelon'], return_tensors='pt', padding=True)
    input_ids = [input['input_ids'], input2['input_ids'], input3['input_ids']]
    attention_mask = [input['attention_mask'], input2['attention_mask'], input3['attention_mask']]

    d_input = tokenizer(
        ['apple', 'apple and banana are good', 'apple and banana and watermelon are good'], 
        return_tensors='pt', 
        padding=True
    )

    out = biencoder.forward(
        q_tokens=input_ids,
        q_masks=attention_mask,
        d_tokens=[d_input['input_ids']],
        d_masks=[d_input['attention_mask']]
    )
    print(out)

def test_reward_wrapper():
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v0.6')
    model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v0.6')

    from reward_wrapper import GenerativeRewardWrapper, Metric
    reward_value_model = GenerativeRewardWrapper(
        generator=model,
        tokenizer=tokenizer,
        utility=Metric('rouge'),
        generation_config=model.generation_config
    )

    q, r, r_texts = reward_value_model._inference(
        ['hello, I am', 'the reason i am here is']
    )

    rw = reward_value_model.get_rewards(["app", "Thank you"], r_texts)
    print(rw)

test_bi_encder()
# test_reward_wrapper()
