import os
import json
import argparse
from collections import defaultdict 
from tqdm import tqdm 
from tools import load_topic, batch_iterator
from pyserini.search.lucene import LuceneSearcher

def search(index, k1, b, topic, batch_size, k, output):

    searcher = LuceneSearcher(index)
    searcher.set_bm25(k1=k1, b=b)

    topics = load_topic(topic)
    qids = list(topics.keys())
    qtexts = list(topics.values())

    for (start, end) in tqdm(
            batch_iterator(range(0, len(qids)), batch_size, True),
            total=(len(qids)//batch_size)+1
    ):
        qids_batch = qids[start: end]
        qtexts_batch = qtexts[start: end]
        hits = searcher.batch_search(
                queries=qtexts_batch, 
                qids=qids_batch, 
                threads=32,
                k=k,
        )

        for key, value in hits.items():
            for i in range(len(hits[key])):
                ## save in memory
                if isinstance(output, dict):
                    output[key].append(hits[key][i].docid)
                ## write out
                else:
                    output.write(
                        f'{key} Q0 {hits[key][i].docid} {i+1} {hits[key][i].score:.5f} bm25\n'
                    )
    return output  # could be a writer or a dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1",type=float, default=0.9) # 0.5 # 0.82
    parser.add_argument("--b", type=float, default=0.4) # 0.3 # 0.68
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--index", default=None, type=str)
    parser.add_argument("--topic", default=None, type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output", default=None, type=str)
    ## reconstruct a new data
    args = parser.parse_args()

    ## search
    if args.output is not None:
        output = open(args.output, 'w')
    else:
        output = defaultdict(list)
    search(
        index=args.index, 
        k1=args.k1, 
        b=args.b, 
        topic=args.topic, 
        batch_size=args.batch_size, 
        k=args.k,
        output=output
    )
    if args.output is not None:
        output.close()

    ## reconsturct

    print('done')
