import os
import json
from datasets import load_dataset

# query and qrels
os.makedirs('/home/dju/datasets/litsearch/qrels', exist_ok=True)
query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
with open('/home/dju/datasets/litsearch/queries.jsonl', 'w') as fq, \
     open('/home/dju/datasets/litsearch/qrels/test.tsv', 'w') as fqrels:

    fqrels.write("query-id\tcorpus-id\tscore\n")

    for i, query in enumerate(query_data):
        query_set = query['query_set']
        specificity = query['specificity']
        quality = query['quality']
        query_id = f"{query_set}_{i}" + f":{specificity}:{quality}"

        fq.write(json.dumps({"_id": query_id, "text": query['query']})+'\n')
        for p in query['corpusids']:
            fqrels.write(f"{query_id}\t{p}\t1\n")


# corpus 
os.makedirs('/home/dju/datasets/litsearch/full_paper', exist_ok=True)
corpus_clean_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
with open('/home/dju/datasets/litsearch/corpus.jsonl', 'w') as fc, \
     open('/home/dju/datasets/litsearch/full_paper/corpus.jsonl', 'w') as fc_full:

    for doc in corpus_clean_data:
        docid = str(doc['corpusid'])
        doctitle = doc['title']
        docabstract = doc['abstract']
        docfull_paper = doc['full_paper']

        fc.write(json.dumps({
            "_id": docid, "title": doctitle, "text": docabstract,
        })+'\n')
        fc_full.write(json.dumps({
            "_id": docid, "title": doctitle, "text": docabstract, "full_paper": docfull_paper
        })+'\n')

# corpus_s2orc_data = load_dataset("princeton-nlp/LitSearch", "corpus_s2orc", split="full")
