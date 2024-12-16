import argparse
import torch
from modeling import SparseEncoder
from modeling.biencoders.sparse_crossattn import SparseAdaptiveEncoders
from beir.datasets.data_loader import GenericDataLoader
from _impact_searcher import LuceneImpactSearcher
from transformers import AutoTokenizer
from modeling.llm import vLLM
import ir_measures
from ir_measures import nDCG, R

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def apply_docs_prompt(doc_items, field='text'):
    doc_prompt_template = "[{ID}]{T}{P}\n"
    p = ""
    for idx, doc_item in enumerate(doc_items):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        title = doc_item.get('title', '')
        if title != '':
            p_doc = p_doc.replace("{T}", f" (Title: {title}) ")
        else:
            p_doc = p_doc.replace("{T}", "")
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p

def prepare_encoder(args):
    encoder = SparseAdaptiveEncoders(
        encoder=SparseEncoder(args.d_encoder_name),
        q_encoder=SparseEncoder(args.q_encoder_name_or_path, cross_attention=True) 
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.d_encoder_name)
    return encoder, tokenizer

@torch.no_grad()
def evaluate(args):
    """ only made for evaluting query encoder, and the documents have already been indexed.  
    Focusing on the ndcg@10 and R@100"""

    # load dataset (qrels)
    corpus, queries, qrels = GenericDataLoader(data_folder=args.dataset_dir).load(split=args.split)
    searcher = LuceneImpactSearcher(args.index_dir, args.d_encoder_name)
    ## [TODO] shuffle or small subset 

    ## load model 
    ada_encoder, tokenizer = prepare_encoder(args)
    generator = vLLM(args.generator_name, num_gpus=1) if args.iteration > 0 else None
    q_ids = list(queries.keys())[:args.debug]

    # inference runs by searching
    runs = {}
    for batch_q_ids in batch_iterator(q_ids, args.batch_size):
        batch_q_texts = [queries[id] for id in batch_q_ids]

        # proces queries and produce reprs.
        q_inputs = tokenizer(
            batch_q_texts,
            add_special_tokens=True,
            max_length=args.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(args.device)
        q_outputs = ada_encoder(q_inputs['input_ids'], q_inputs['attention_mask']) 
        q_reps = q_outputs.reps

        ## iterative search -- 1
        hits = searcher.batch_search(
            logits=q_reps.float().detach().cpu().numpy(), 
            q_ids=batch_q_ids,
            k=100,
            threads=32
        )

        ## iterative search > 1
        if args.iteration > 0:

            ### prepare LLMPRF
            prf_prompts = []
            for query, hit in zip(batch_q_texts, hits):
                docs = apply_docs_prompt([h.docid for h in hit[args.top_k]])
                prf_prompts.append(prompt.format(query, docs))
            prf = generator.generate(prf_prompts)

            ### process feedbacks and queries and produce reprs.
            f_inputs = tokenizer(
                prf,
                add_special_tokens=True,
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(args.device)
            q_outputs = ada_encoder(None, None, 
                f_inputs['input_ids'], f_inputs['attention_mask'],
                prev_output=q_outputs.prev_out
            )
            q_reps = q_outputs.reps

            hits = searcher.batch_search(
                logits=q_reps.float().detach().cpu().numpy(), 
                q_ids=batch_q_ids,
                k=100,
                threads=32
            )

        for id in batch_q_ids:
            batch_runs = {h.docid: h.score for h in hits[id]}
            # print([h.docid for h in hits[id]])
            runs.update({id: batch_runs})

    # load run
    result = ir_measures.calc_aggregate([nDCG@10, R@100], qrels, runs) 
    # print(result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",type=str, default=None)
    parser.add_argument("--index_dir",type=str, default=None)
    parser.add_argument("--d_encoder_name", type=str, default='naver/splade-v3')
    parser.add_argument("--q_encoder_name_or_path", type=str, default='naver/splade-v3')
    parser.add_argument("--generator_name", type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument("--split", type=str, default='train') 
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--iteration", type=int, default=0) 
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default='cuda') 
    parser.add_argument("--debug", type=int, default=None)
    args = parser.parse_args()

    prompt = "Rewrite the question with more comprehensive contexts, making the question easier to understand. Some useful knowledge could be found in the given texts from search engine (but some of which might be irrelevant).\n\nQuestion: {}\nContexts:\n{}\nRewritten question:\n"

    evaluate(args)
