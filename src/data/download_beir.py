import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from datasets import load_dataset

    # 'msmarco', 'nfcorpus', 'nq', \
dataset_list = [
    'nq-train', \
    'hotpotqa', 'fiqa', 'cqadupstack', 'quora', \
    'dbpedia-entity', 'fever', 'scifact'
]
dataset_list_testonly = [
    'trec-covid', 'arguana', 'webis-touche2020', 'scidocs', 'climate-fever'
]

DATASET_ROOT='/home/dju/datasets/beir-cellar/'
# DATASET_ROOT='/home/dju/datasets/beir-new/'
os.makedirs(DATASET_ROOT, exist_ok=True)

# queries and corpus
for dataset_name in dataset_list_testonly:
    # original site
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    data_path = util.download_and_unzip(url, DATASET_ROOT)

    # huggingface
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
