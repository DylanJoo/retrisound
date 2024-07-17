from collections import defaultdict
import json
# from tqdm import tqdm

def load_corpus_file(file):
    corpus = {}
    with open(file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            docid = item['id']
            corpus[docid] = {
                'text': item['meta']['content'],
                'title': item['meta']['title']
            }
    return corpus

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

# def load_corpus_file_v2(file, corpus):
#
#     with open(file, 'r') as f:
#         for line in f:
#             item = json.loads(line.strip())
#             docid = item['id']
#             text = item['meta']['content']
#             title = item['meta']['title']
#             try:
#                 corpus[docid] = {'text': text, 'title': title}
#     return corpus


