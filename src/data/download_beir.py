import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

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
os.makedirs(DATASET_ROOT, exist_ok=True)
for dataset_name in dataset_list:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    # out_dir = os.path.join(DATASET_ROOT, dataset_name)
    data_path = util.download_and_unzip(url, DATASET_ROOT)
    print(data_path)
    # corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
