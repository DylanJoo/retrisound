from __future__ import annotations

import csv
import json
import logging
import os
import ir_datasets
from collections import defaultdict

from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


class IRDataLoader:
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            self.prefix = prefix 
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load_from_ir_datasets(
        self, ignore_corpus=False
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        dataset = ir_datasets.load(self.prefix)

        # Corpus
        if ignore_corpus:
            # self.corpus = defaultdict(lambda: {'title': "", 'text': ''})
            self.corpus = None
        else:
            logger.info("Loading Corpus...")
            for doc in dataset.docs_iter():
                self.corpus[doc.doc_id] = {
                    "text": doc.text if hasattr(doc, "text") else doc.body,
                    "title": doc.title if hasattr(doc, "title") else "",
                }
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        # Queries
        logger.info("Loading Queries...")
        for query in dataset.queries_iter():
            self.queries[query.query_id] = query.text if hasattr(query, "text") else query.description

        # Qrels
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in self.qrels:
                self.qrels[qrel.query_id] = {qrel.doc_id: qrel.relevance}
            else:
                self.qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        logger.info("loaded %d queries.", len(self.queries))
        logger.info("query example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(
        self, split="test", ignore_corpus=False
    ) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if ignore_corpus:
            self.corpus = None
        else:
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("loaded %d %s queries.", len(self.queries), split.upper())
            logger.info("query example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> dict[str, dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                id_field = "_id" if "_id" in line else "id"
                self.corpus[line[id_field]] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):

        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):
        reader = csv.reader(
            open(self.qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
