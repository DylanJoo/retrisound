import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from base_encoder import Contriever
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Literal

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = Contriever.from_pretrained('bert-base-uncased', pooling='mean')
model_cls = Contriever.from_pretrained('OpenMatch/cocodr-base-msmarco', pooling='cls')

@dataclass
class ModelOptions:
    retriever_name_or_path: Optional[str] = field(default="facebook/contriever")
    add_pooling_layer: Optional[bool] = field(default=False)
    num_mem_tokens: Optional[int] = field(default=1)
    num_budget: Optional[int] = field(default=5)
    tau: Optional[float] = field(default=1.0)

def test_q_encoder():
    from base_encoder import Contriever
    q_encoder = Contriever.from_pretrained('OpenMatch/cocodr-base-msmarco', pooling='cls')
    input = tokenizer(['hello world', 'apple'], return_tensors='pt', padding=True)
    out = q_encoder(**input)
    return q_encoder

def test_crossencoder():
    from crossencoder import ValueCrossEncoder
    from transformers import BertForSequenceClassification

    model_opt = ModelOptions()
    cross_encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    crossencoder = ValueCrossEncoder(
        model_opt,
        cross_encoder=cross_encoder,
        d_encoder=model,
        n_max_candidates=10
    )

    outputs = test_biencder()
    out = crossencoder.forward(
        qembs=outputs.qembs, 
        dembs=outputs.dembs
    )
    print(out)

def test_biencder():
    # modification
    from modifier import FeedbackQueryModifier
    model_opt = ModelOptions()
    encoder = test_q_encoder()

    biencoder = FeedbackQueryModifier(
        model_opt, 
        qr_encoder=encoder, 
        qf_encoder=encoder,
        d_encoder=encoder,
    )

    # query
    input = tokenizer(['apple', 'apple2', 'apple3'], return_tensors='pt', padding=True)
    input2 = tokenizer(['banana', 'banana2', 'banana3'], return_tensors='pt', padding=True)
    input3 = tokenizer(['watermelon', 'watermelon2', 'watermelon3'], return_tensors='pt', padding=True)
    input4 = tokenizer(['aaa', 'bbb', 'ccc'], return_tensors='pt', padding=True)

    # documents
    d_input = tokenizer(
        ['apple', 'apple and banana are good', 'apple and banana and watermelon are good'], 
        return_tensors='pt', 
        padding=True
    )

    # B = 3 | N = 4 | M = 2
    out = biencoder.forward(
        q_tokens=[input['input_ids'], input2['input_ids'], input3['input_ids'], input4['input_ids']],
        q_masks=[input['attention_mask'], input2['attention_mask'], input3['attention_mask'], input4['input_ids']],
        d_tokens=[d_input['input_ids'], d_input['input_ids']],
        d_masks=[d_input['attention_mask'], d_input['attention_mask']],
    )
    print(out.qembs.shape)
    print(out.dembs.shape)
    print(out.scores.shape)
    print(out.loss.shape)
    return out

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

# test_biencder()
test_crossencoder()
# test_reward_wrapper()
