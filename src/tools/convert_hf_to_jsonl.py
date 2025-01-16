import os
import json
from datasets import load_dataset

DATASET_DIR='/ivi/ilps/personal/dju/datasets/lit-search'

os.makedirs(DATASET_DIR, exist_ok=True)
corpus_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
# corpus_data = corpus_data.map(lambda x: {"contents": x["title"] + " " + x["abstract"]})


with open(os.path.join(DATASET_DIR, 'corpus_data.jsonl'), 'w') as f:
    for i in range(len(corpus_data)):
        data = corpus_data[i]
        f.write(json.dumps({
            "id": data["corpusid"], 
            "contents": data["abstract"],
            "title": data['title']
        })+'\n')

