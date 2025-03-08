import csv
import json
import logging
import os
from tqdm import tqdm
from glob import glob
from data.utils import load_corpus_file, batch_iterator
from datasets import load_dataset
from collections import defaultdict

logger = logging.getLogger(__name__)


class QADataLoader:
    def __init__(
        self,
        dataset_dir: str,
        split: str = 'train',
        corpus_path: str = None,
        run_file: str = None
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.corpus_path = (corpus_path or '/home/dju/datasets/wikipedia_split_dpr_shards')

        self.corpus = defaultdict(lambda: {"text": "", "title": ""})

        self.questions = {}
        self.answers = {}
        self.qrels = {} # in QA datasets, qrels mean the short answer 

    def load(self, split='train'):
        if 'asqa' in self.dataset_dir.lower():
            return self.load_asqa(split)
        if '2WikiMultiHop'.lower() in self.dataset_dir.lower():
            return self.load_2wikimultihop(split)
        else:
            raise NotImplementedError

    def _load_corpus(self, path):
        if not os.path.isdir(path):
            self.corpus = load_corpus_file(path)
        else:
            from multiprocessing import Pool
            files = glob(f'{path}/*jsonl*') # remove this after debugging

            for batch_files in tqdm(batch_iterator(files, 1000), 'Load wiki files', total=1+len(files)//1000):
                with Pool(processes=16) as pool:
                    corpora = pool.map(load_corpus_file, batch_files)

                for corpus in corpora:
                    for docid, docdict in corpus.items():
                        self.corpus[docid] = docdict
                del corpora

    def load_asqa(self, split):
        data_file = os.path.join(self.dataset_dir, "ASQA.json")
        raw_data = json.load(open(data_file, 'r'))[split]

        for id in raw_data:
            logger.info("Loading Questions and Answers...")
            self.questions[id] = raw_data[id]['ambiguous_question']
            self.answers[id] = {"text": raw_data[id]['annotations'][0]['long_answer'], "title": ""}
            self.qrels[id] = [pair['short_answers'] for pair in raw_data[id]['qa_pairs']]

        self._load_corpus(self.corpus_path)
        return self.corpus, self.questions, self.answers

    def load_2wikimultihop(self, split):
        data = load_dataset('xanhho/2WikiMultihopQA', split=split)

        for i, id in enumerate(data['_id']):
            logger.info("Loading Questions and Answers...")
            self.questions[id] = data[i]['question']
            self.qrels[id] = [title for title in data[i]['supporting_facts']['title']]
            self.answers[id] = {"text": data[i]['answer'], "title": " ".join(self.qrels[id])}

        self._load_corpus(self.corpus_path)
        return self.corpus, self.questions, self.answers


    # def _load_runs(self, file, negative_threshold=50):
    #     with open(file, 'r') as f:
    #         for line in f:
    #             qid, _, docid, rank, score, _ = line.strip().split()
    #             if rank == 1:
    #                 self.qrels[qid] = {docid: 1}
    #             if rank >= 50:
    #                 self.qrels[qid].update({docid: 0})
