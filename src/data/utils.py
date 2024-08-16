from collections import defaultdict
import json
from tqdm import tqdm

def load_corpus_file(file):
    corpus = {}
    with open(file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            docid = item['id']
            # the qampari's wiki pasages have the key 'meta'
            if 'meta' in item.keys():
                corpus[docid] = {
                    'text': item['meta']['content'],
                    'title': item['meta']['title']
                }
            else:
                corpus[docid] = {
                    'text': item['contents'],
                    'title': item['title']
                }
    return corpus

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def convert_tsv_to_jsonl(file):
    writer = open(file.replace('tsv', 'jsonl'), 'w')
    with open(file, 'r') as f:
        firstline = f.readline()
        for line in tqdm(f):
            row = line.strip().split('\t')
            writer.write(json.dumps({
                "id": f"wiki:{row[0]}",
                "contents": f"{row[2]} {row[1]}",
                'title': row[2]
            })+'\n')
    writer.close()
