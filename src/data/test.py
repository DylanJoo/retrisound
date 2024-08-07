from qampari import ContextQADataset


split = 'train'
dataset = ContextQADataset(
    data_file=f'/home/dju/datasets/qampari/{split}_data.jsonl',
    n_max_segments=10,
    n_max_candidates=50,
    budget=5,
    depth=10,
    corpus_file='/home/dju/datasets/qampari/wikipedia_chunks/chunks_v5',
    retrieval_file=f'/home/dju/datasets/qampari/{split}_data_bm25-top100.run',
)
# add action for index 0
dataset.add_feedback(0, 'this is a testing action')
# print(dataset[0])

features = [dataset[i] for i in range(4)]
print(features)

# from qampari import ContextQACollator
# from transformers import AutoTokenizer
# tokenizer_r = AutoTokenizer.from_pretrained('bert-base-uncased')
# tokenizer_g = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
# collator = ContextQACollator(
#     tokenizer_r=tokenizer_r,
#     tokenizer_g=tokenizer_g
# )
#
# d=collator(features)
# print(d.keys())
