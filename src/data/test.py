from qampari import ContextQADataset

split = 'test'
dataset = ContextQADataset(
    data_file=f'/home/dju/datasets/qampari/{split}_data.jsonl',
    n_max_segments=10,
    n_max_candidates=50,
    budget=5,
    depth=10,
    corpus_file='/home/dju/datasets/qampari/wikipedia_chunks/chunks_v5',
    retrieval_file=f'/home/dju/datasets/qampari/{split}_data_bm25-top100.run',
    quick_test=5
)
# add action for index 0
dataset.add_action(0, 'this is 0 testing action (next)')
# print(dataset[0])

from qampari import ContextQACollator
from transformers import AutoTokenizer
tokenizer_r = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_g = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
collator = ContextQACollator(
    tokenizer_r=tokenizer_r,
    tokenizer_g=tokenizer_g
)
features = [dataset[i] for i in range(4)]
d=collator(features)
print(d['inputs_for_retriever'].keys())

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)

for b in loader:
    print(b['inputs_for_retriever']['q_tokens'])
    dataset.add_action(0, f'this is 0 testing action')
    dataset.add_action(1, f'this is 1 testing action')
    dataset.add_action(2, f'this is 2 testing action')
    dataset.add_action(3, f'this is 3 testing action')
    dataset.add_action(4, f'this is 4 testing action')
    dataset.add_action(5, f'this is 5 testing action')

