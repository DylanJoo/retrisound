from collections import defaultdict
import json

def load_corpus_file(file):
    corpus = defaultdict(dict)
    with open(file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            docid = item['id']
            corpus[docid]['text'] = item['meta']['content']
            corpus[docid]['title'] = item['meta']['title']
    return corpus

