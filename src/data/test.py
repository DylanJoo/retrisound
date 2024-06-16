from qampari import ContextQADataset


dataset = ContextQADataset(
    data_file='/home/dju/datasets/qampari/test_data.jsonl',
    n_max_segments=10,
    corpus_file=None,
    budget=10
)
dataset.add_action(0, 'this is a testing action')
print(dataset[0])

features = [dataset[i] for i in range(4)]
from qampari import ContextQACollator
from transformers import AutoTokenizer
tokenizer_r = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_g = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
collator = ContextQACollator(
    tokenizer_r=tokenizer_r,
    tokenizer_g=tokenizer_g
)

d=collator(features)
print(d.keys())

