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
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
collator = ContextQACollator(tokenizer=tokenizer)

d=collator(features)
print(d)

