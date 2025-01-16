# # from qampari import ContextQADataset
# from asqa import ContextQADataset
#
# split = 'test'
# dataset = ContextQADataset(
#     data_file=f'/home/dju/datasets/asqa/ASQA.json',
#     n_max_segments=5,
#     n_max_candidates=50,
#     budget=5,
#     depth=10,
#     corpus_file='/home/dju/datasets/wikipedia_split',
#     retrieval_file=f'/home/dju/datasets/asqa/train_data_bm25-top100.run',
# )
# # add action for index 0
# dataset.add_feedback(0, 'this is 0 testing action (next)')
# exit(0)
#
# from asqa import ContextQACollator
# from transformers import AutoTokenizer
# tokenizer_r = AutoTokenizer.from_pretrained('bert-base-uncased')
# tokenizer_g = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
# collator = ContextQACollator(
#     tokenizer_r=tokenizer_r,
#     tokenizer_g=tokenizer_g
# )
# features = [dataset[i] for i in range(4)]
# d=collator(features)
# print(d['inputs_for_retriever'].keys())
# print(d['candidates'])
#

from utils import load_corpus_file, batch_iterator
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
def load_corpus():
    all_corpora = {}
    files = glob(f'/home/dju/datasets/wikipedia_split/*jsonl*')

    for batch_files in tqdm(batch_iterator(files, 1000), 'load wiki files', total=1+len(files)//1000):
        with Pool(processes=16) as pool:
            corpora = pool.map(load_corpus_file, batch_files)

        for corpus in corpora:
            for docid, docdict in corpus.items():
                all_corpora[docid] = docdict
        del corpora
    return all_corpora
