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
tokenizer_r = AutoTokenizer.from_pretrained(R_model_name_or_path)
from modeling.rmt import RMTEncoder
from modeling.rife import Contriever
model = Contriever.from_pretrained(R_model_name_or_path)
encoder = deepcopy(model)
ada_encoder = RMTEncoder(
        base_model=model, 
        num_mem_tokens=4,
        tokenizer=tokenizer_r,
        input_size=512,
        sum_loss=False
)
from modeling import InBatchInteraction
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

tokenizer_g = AutoTokenizer.from_pretrained(G_model_name_or_path)
from modeling import RerankAugmentedGeneration
model = RerankAugmentedGeneration(
        llm=generator, 
        tokenizer=tokenizer_g,
        biencoders=bi_encoders,
)

## add data
split='test'
from data.qampari import ContextQADataset
dataset = ContextQADataset(
    data_file=f'/home/dju/datasets/qampari/{split}_data.jsonl',
    n_max_segments=10,
    n_max_candidates=50,
    budget=5,
    depth=10,
    corpus_file=None,
    retrieval_file=f'/home/dju/datasets/qampari/{split}_data_bm25-top100.run',
    quick_test=True
)
    # corpus_file='/home/dju/datasets/qampari/wikipedia_chunks/chunks_v5',
dataset.add_action(0, 'this is a testing action')

features = [dataset[i] for i in range(4)]
from data.qampari import ContextQACollator
collator = ContextQACollator(
    tokenizer_r=tokenizer_r,
    tokenizer_g=tokenizer_g
)
d=collator(features)
print(d['inputs_for_retriever'])
o=model(**d)
print(o.loss_g)
