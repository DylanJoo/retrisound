import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import re
import argparse
import torch
import torch.nn.functional as F
from data.ir_dataloader import IRDataLoader
from _impact_searcher import LuceneImpactSearcher
from transformers import AutoTokenizer
from modeling.llm.vllm_api import LLM
import ir_measures
from ir_measures import nDCG, R
from tqdm import tqdm
import pickle

def transform_ids_to_vector(inputs, tokenizer, count=False):
    vector = torch.zeros(inputs.size(0), tokenizer.vocab_size).to(inputs.device)
    if count:
        vector = vector.scatter_add(1, inputs, torch.ones_like(inputs, dtype=vector.dtype))
    else:
        vector = vector.scatter(1, inputs, 1)
    return vector

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
    from modeling.encoder import SparseEncoderExp
    q_encoder = SparseEncoderExp.from_pretrained(
        args.q_encoder_name_or_path,
        add_cross_attention=False, is_decoder=False, 
        num_hidden_layers=1,
        num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(args.d_encoder_name)
    return q_encoder, tokenizer

def postprocess_output(output, tag='p'):
    output = output.split(f'</{tag}>')[0]
    output = output.split('Query:')[0]
    output = output.split('\n')[0]
    output = re.sub(r"\d+\.\s", "", output).strip()
    output = re.sub(r"-\s", "", output).strip()
    return output

from prompts.generic import apply_report_inst_prompt, apply_report_inst_prompt_zs, apply_rewrite_inst_prompt_zs

prompt = {"qr": apply_rewrite_inst_prompt_zs, "rg": apply_report_inst_prompt_zs, "prf-rg": apply_report_inst_prompt}

@torch.no_grad()
def evaluate(args):
    """ only made for evaluting query encoder, and the documents have already been indexed.  
    Focusing on the ndcg@10 and R@100 """

    # load dataset (qrels)
    if os.path.exists(args.dataset_dir):
        corpus, queries, qrels = IRDataLoader(data_folder=args.dataset_dir).load(split=args.split)
    else: 
        corpus, queries, qrels = IRDataLoader(prefix=args.dataset_dir).load_from_ir_datasets(
            ignore_corpus=False
        )
    searcher = LuceneImpactSearcher(args.index_dir, args.d_encoder_name)
    ## [TODO] shuffle or small subset 

    ## load model 
    _, tokenizer = prepare_encoder(args) 

    q_ids = list(queries.keys())[:args.debug]

    ## load generators
    generator = LLM(args.generator_name, num_gpus=1, gpu_memory_utilization=0.9) if args.iteration > 0 else None

    # inference runs by searching
    runs = {}
    for batch_q_ids in tqdm(batch_iterator(q_ids, args.batch_size), total=len(q_ids)//args.batch_size + 1):
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
        q_reps = transform_ids_to_vector(q_inputs.input_ids, tokenizer, count=args.count)

        ## iterative search -- 1
        hits = searcher.batch_search(
            logits=q_reps.float().detach().cpu().numpy(), 
            q_ids=batch_q_ids,
            k=100,
            threads=32
        )

        ## iterative search > 1
        for iter in range(1, args.iteration+1):

            ### prepare LLMPRF
            prf_prompts = []
            for query, id in zip(batch_q_texts, hits):
                docs = apply_docs_prompt([corpus[h.docid] for h in hits[id][:args.top_k]])
                prf_prompts.append(prompt[args.prompt_type](query, docs))

            prf = generator.generate(prf_prompts, max_tokens=128, min_tokens=0)
            if args.prompt_type == 'qr':
                batch_o_texts = [postprocess_output(o, tag='q') for o in prf]

            if args.prompt_type == 'rg':
                batch_o_texts = [postprocess_output(o, tag='p') for o in prf]

            for o, q in zip(batch_o_texts, batch_q_texts):
                print(f"# {q} --> {o}\n")

            ### process feedbacks and queries and produce reprs.
            f_inputs = tokenizer(
                [(q * args.repeat_query + " " + o).strip() for (q, o) in zip(batch_q_texts, batch_o_texts)],
                add_special_tokens=True,
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(args.device)

            if args.adaptive:
                q_reps = self.q_encoder(
                    input_ids=f_inpus['input_ids'],
                    attention_mask=f_inputs['attention_mask'],
                )
            else:
                q_reps = transform_ids_to_vector(f_inputs['input_ids'], tokenizer, count=args.count)
            
            ### Re-retrieve
            hits = searcher.batch_search(
                logits=q_reps.float().detach().cpu().numpy(), 
                q_ids=batch_q_ids,
                k=100,
                threads=32
            )

        # convert to runs
        for id in batch_q_ids:
            try:
                batch_runs = {h.docid: h.score for h in hits[id]}
            except:
                logger.warn(f"No relevant documents found for query {id}")
                batch_runs = {'NA': 0}
            runs.update({id: batch_runs})

    # load run
    result = ir_measures.calc_aggregate([nDCG@10, nDCG@20, R@100], qrels, runs) 
    return (runs, (result[nDCG@10], result[nDCG@20], result[R@100]) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",type=str, default=None)
    parser.add_argument("--index_dir",type=str, default=None)
    parser.add_argument("--d_encoder_name", type=str, default='naver/splade-v3')
    parser.add_argument("--q_encoder_name_or_path", type=str, default='naver/splade-v3')
    parser.add_argument("--split", type=str, default='train') 

    ## generation-augmentation
    parser.add_argument("--generator_name", type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--iteration", type=int, default=0) 
    parser.add_argument("--repeat_query", type=int, default=0)
    parser.add_argument("--prompt_type", type=str, default='qr')
    parser.add_argument("--adaptive", action='store_true', default=False) # leaned or zero-shot
    parser.add_argument("--count", action='store_true', default=False)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_pickle", action='store_true', default=False)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda') 
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--exp", type=str, default='debug')
    args = parser.parse_args()

    runs, results = evaluate(args)
    print(f" ============= ")
    print(f" [Data path] {args.dataset_dir}")
    print(f" [Rd] {args.d_encoder_name} [Rq] {args.q_encoder_name_or_path} [G] {args.generator_name}")
    print(f"### {args.dataset_dir.split('/')[-1]} | {args.exp} | {results[0]:.4f} |")
    print(f" ============= ")

    if args.save_pickle:
        with open( f'run-{args.exp}.pickle', 'wb') as f:
            pickle.dump(runs, f, pickle.HIGHEST_PROTOCOL)
