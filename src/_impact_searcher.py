from pyserini.search.lucene import LuceneImpactSearcher as _LuceneImpactSearcher
import logging
import os
import pickle
from collections import namedtuple
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import scipy
from tqdm import tqdm

from pyserini.index.lucene import Document, IndexReader
from pyserini.pyclass import autoclass, JFloat, JInt, JArrayList, JHashMap
from pyserini.search.lucene import JScoredDoc
from pyserini.util import download_prebuilt_index, download_encoded_corpus
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Wrappers around Anserini classes
JSimpleImpactSearcher = autoclass('io.anserini.search.SimpleImpactSearcher')


class LuceneImpactSearcher(_LuceneImpactSearcher):
    """ encoder_type: `pytorch`. This can be a dummy encoder class, as we will provide the pre-encoded logits"""

    def __init__(
        self, 
        index_dir: str, 
        query_encoder: str, 
        min_idf=0, 
        prebuilt_index_name=None,
        device='cpu'
    ):
        self.index_dir = index_dir
        self.idf = self._compute_idf(index_dir)
        self.min_idf = min_idf
        self.object = JSimpleImpactSearcher(index_dir)
        self.num_docs = self.object.get_total_num_docs()
        self.encoder_type = 'pytorch'
        self.prebuilt_index_name = prebuilt_index_name

        # encoder attributres
        self.tokenizer = AutoTokenizer.from_pretrained(query_encoder or 'naver/splade-v3')
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        self.weight_range = 5
        self.quant_range = 256

    def encode(self, text=None, batch_aggregated_logits=None, max_length=256, **kwargs):
        if batch_aggregated_logits is None:
            inputs = self.tokenizer([text], max_length=max_length, padding='longest',
                                    truncation=True, add_special_tokens=True,
                                    return_tensors='pt').to(self.device)
            input_ids = inputs['input_ids']
            input_attention = inputs['attention_mask']
            batch_logits = self.query_encoder(input_ids)['logits']
            batch_aggregated_logits, _ = torch.max(torch.log(1 + torch.relu(batch_logits))
                                                   * input_attention.unsqueeze(-1), dim=1)
            batch_aggregated_logits = batch_aggregated_logits.cpu().detach().numpy() # done previously

        raw_weights = self._output_to_weight_dicts(batch_aggregated_logits)
        return self._get_encoded_query_token_wight_dicts(raw_weights)[0]

    def _output_to_weight_dicts(self, batch_aggregated_logits):
        to_return = []
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            d = {self.reverse_voc[k]: float(v) for k, v in zip(list(col), list(weights))}
            to_return.append(d)
        return to_return

    def _get_encoded_query_token_wight_dicts(self, tok_weights):
        to_return = []
        for _tok_weight in tok_weights:
            _weights = {}
            for token, weight in _tok_weight.items():
                weight_quanted = round(weight / self.weight_range * self.quant_range)
                _weights[token] = weight_quanted
            to_return.append(_weights)
        return to_return

    def batch_search(self, 
                     logits: torch.FloatTensor = None, 
                     q_ids: List[str] = None,
                     k: int = 10, threads: int = 1, fields=dict()) -> Dict[str, List[JScoredDoc]]:
        """Search the collection concurrently for multiple queries, using multiple threads.

        Parameters
        ----------
        logits: List[str]
            List of query string/vector.
        q_ids : List[str]
            List of corresponding query ids.
        k : int
            Number of hits to return.
        threads : int
            Maximum number of threads to use.
        fields : dict
            Optional map of fields to search with associated boosts.

        Returns
        -------
        Dict[str, List[JScoredDoc]]
            Dictionary holding the search results, with the query ids as keys and 
            the corresponding lists of search results as the values.
        """
        query_lst = JArrayList()
        qid_lst = JArrayList()
        for logit in logits:
            jquery = JHashMap()
            if self.encoder_type == 'pytorch':
                logit = np.expand_dims(logit, axis=0)
                encoded_query = self.encode(text=None, batch_aggregated_logits=logit)
                for (token, weight) in encoded_query.items():
                    if token in self.idf and self.idf[token] > self.min_idf:
                        jquery.put(token, JInt(weight))
            else:
                jquery = q
            query_lst.add(jquery)

        for qid in q_ids:
            jqid = qid
            qid_lst.add(jqid)

        jfields = JHashMap()
        for (field, boost) in fields.items():
            jfields.put(field, JFloat(boost))

        if not fields:
            if self.encoder_type == 'onnx':
                results = self.object.batch_search_queries(query_lst, qid_lst, int(k), int(threads))
            else:
                results = self.object.batch_search(query_lst, qid_lst, int(k), int(threads))
        else:
            results = self.object.batch_search_fields(query_lst, qid_lst, int(k), int(threads), jfields)
        return {r.getKey(): r.getValue() for r in results.entrySet().toArray()}

    def doc(self, docid: Union[str, int]) -> Optional[Document]:
        """Return the :class:`Document` corresponding to ``docid``. The ``docid`` is overloaded: if it is of type
        ``str``, it is treated as an external collection ``docid``; if it is of type ``int``, it is treated as an
        internal Lucene ``docid``. Method returns ``None`` if the ``docid`` does not exist in the index.

        Parameters
        ----------
        docid : Union[str, int]
            Overloaded ``docid``: either an external collection ``docid`` (``str``) or an internal Lucene ``docid``
            (``int``).

        Returns
        -------
        Document
            :class:`Document` corresponding to the ``docid``.
        """
        lucene_document = self.object.doc(docid)
        if lucene_document is None:
            return None
        return Document(lucene_document)

    def doc_by_field(self, field: str, q: str) -> Optional[Document]:
        """Return the :class:`Document` based on a ``field`` with ``id``. For example, this method can be used to fetch
        document based on alternative primary keys that have been indexed, such as an article's DOI. Method returns
        ``None`` if no such document exists.

        Parameters
        ----------
        field : str
            Field to look up.
        q : str
            Unique id of document.

        Returns
        -------
        Document
            :class:`Document` whose ``field`` is ``id``.
        """
        lucene_document = self.object.doc_by_field(field, q)
        if lucene_document is None:
            return None
        return Document(lucene_document)

    def close(self):
        """Close the searcher."""
        self.object.close()

    # def search(self, logit: torch.FloatTensor = None, k: int = 10, fields=dict()) -> List[JScoredDoc]:
    #     """Search the collection.
    #
    #     Parameters
    #     ----------
    #     logit : torch.FloatTensor
    #         Query sparse vector.
    #     k : int
    #         Number of hits to return.
    #     fields : dict
    #         Optional map of fields to search with associated boosts.
    #
    #     Returns
    #     -------
    #     List[JScoredDoc]
    #         List of search results.
    #     """
    #
    #     jfields = JHashMap()
    #     for (field, boost) in fields.items():
    #         jfields.put(field, JFloat(boost))
    #
    #     if self.encoder_type == 'pytorch':
    #         jquery = JHashMap()
    #         encoded_query = self.encode(text=None, batch_aggregated_logits=logit)
    #         for (token, weight) in encoded_query.items():
    #             if token in self.idf and self.idf[token] > self.min_idf:
    #                 jquery.put(token, JInt(weight))
    #     else:
    #         jquery = q
    #
    #     if not fields:
    #         hits = self.object.search(jquery, k)
    #     else:
    #         hits = self.object.searchFields(jquery, jfields, k)
    #
    #     return hits
