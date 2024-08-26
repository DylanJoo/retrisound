import collections 
import json
from tqdm import tqdm
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder

def load_searcher(path, dense=False):
    if dense:
        searcher = LuceneSearcher(path)
        searcher.set_bm25(k1=0.9, b=0.4)
    else:
        searcher FaissSearcher(path, None)
    return searcher

def load_runs(path, output_score=False): # support .trec file only
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            run_dict[qid] += [(docid, float(rank), float(score))]

    sorted_run_dict = collections.OrderedDict()
    for qid, docid_ranks in run_dict.items():
        sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_rank, rel_score in sorted_docid_ranks]
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]

    return sorted_run_dict

def load_topic(path, use_answer=False):
    topic = {}
    with open(path) as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            qid = data['qid']
            qtext = data['question_text']
            topic[qid.strip()] = qtext.strip()

    return topic

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

