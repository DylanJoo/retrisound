import torch.nn as nn
import torch
import numpy as np
from transformers import GenerationConfig
from prompts import asqa, qampari
from transformers import AutoTokenizer

def init_generation_config(model_opt, tokenizer):
    stop = ["<|eot_id|>", "ĊĊĊ", "ĊĊ", "<0x0A>", "<|end_of_text|>"]
    stop_token_ids = [tokenizer.eos_token_id] + \
        [tokenizer.convert_tokens_to_ids(token) for token in stop]
    stop_token_ids = list(set(
        [token_id for token_id in stop_token_ids if token_id is not None]
    ))  
    return GenerationConfig(
        do_sample=True,
        temperature=0.5,
        top_p=1.0,
        max_new_tokens=model_opt.max_new_tokens,
        num_return_sequences=1,
        eos_token_id=stop_token_ids
    )

def update_tokenizer(tokenizer, pad_token='[PAD]'):
    """
    pad_token [str]: default is the preserved token of llama3. 
        This should be re-considered if trying to fine-tune.
    add_special_tokens [dict]: optional additional tokens.
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_token})
    else:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def augmentation_response(
    questions, 
    candidates, 
    n_context,
    rankings=None,
    dataset_prefix='asqa',
    answers=None,
    independent=False
):
    ## loading dependencies
    if 'asqa' in dataset_prefix:
        apply_docs_prompt = asqa.apply_docs_prompt
        apply_rsp_inst_prompt = asqa.apply_rsp_inst_prompt
        instruction_prompt = asqa.instruction_prompt

    if 'qampari' in dataset_prefix:
        apply_docs_prompt = qampari.apply_docs_prompt
        apply_rsp_inst_prompt = qampari.apply_rsp_inst_prompt_new
        instruction_prompt = qampari.instruction_prompt_new

    # prepare prompts
    prompts = []
    if independent:
        for j in range(len(candidates)):
            d = apply_docs_prompt([candidates[j]], field='text')
            prompt = apply_rsp_inst_prompt(
                Q=questions, 
                D=d,
                instruction=instruction_prompt,
                A=answers,
            )
            prompts.append(prompt)
    else:
        contexts = [clist[:n_context] for clist in candidates] 
        for i in range(len(questions)):
            D = apply_docs_prompt(contexts[i], field='text')
            prompt = apply_rsp_inst_prompt(
                Q=questions[i], 
                D=D,
                instruction=instruction_prompt,
                A=answers[i],
            )
            prompts.append(prompt)

    return prompts

def augmentation_feedback(
    questions, 
    candidates, 
    n_context, 
    rankings=None,
    dataset_prefix='asqa'
):
    # prepare contexts
    if rankings is not None:
        assert len(candidates) == rankings.size(0)
        contexts = []
        for i in range(len(rankings)):
            reranked_context = [candidates[i][j] for j in rankings[i]]
            contexts.append(reranked_context[:n_context])
    else:
        contexts = [clist[:n_context] for clist in candidates] 

    ## loading dependencies
    if 'asqa' in dataset_prefix:
        apply_docs_prompt = asqa.apply_docs_prompt
        apply_fbk_inst_prompt = asqa.apply_fbk_inst_prompt
        fbk_instruction_prompt = asqa.fbk_instruction_prompt

    if 'qampari' in dataset_prefix:
        apply_docs_prompt = qampari.apply_docs_prompt
        apply_fbk_inst_prompt = qampari.apply_fbk_inst_prompt
        fbk_instruction_prompt = qampari.fbk_instruction_prompt

    # prepare prompts
    prompts = []

    for i in range(len(questions)):
        ## for answering
        D = apply_docs_prompt(contexts[i], field='text')
        prompt = apply_fbk_inst_prompt(
            Q=questions[i], 
            D=D,
            instruction=fbk_instruction_prompt,
        )
            # prefix='Follow-up query:\n<q>'
        prompts.append(prompt)

    return prompts

def get_mini_batch_dict(retriever_inputs, mb_inds):
    mb_retriever_inputs = {}
    for key, list_of_item in retriever_inputs.items():
        mb_retriever_inputs[key] = []
        for item in list_of_item:
            mb_retriever_inputs[key].append(item[mb_inds])
    return mb_retriever_inputs

def convert_texts_to_tensors(texts, tokenizer):
    tensors = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    tensors = tensors['input_ids']
    return tensors

def get_entropy(probs):
    return -torch.sum(torch.log(probs + 1e-10) * probs, dim=-1)

def multiple_sample_and_log_probability(
    scores, 
    sample_size, 
    return_prob=True, 
    batch=False,
    sort=False,
    baseline=False,
    tau=1
):
    if not batch:
        assert scores.dim() == 1
        subtracts = scores.new_zeros((sample_size, scores.size(0)))
        batch_index = torch.arange(sample_size, device=scores.device)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(0)):
            probs = nn.functional.softmax( (scores - subtracts)/tau, dim=1) + 1e-10
            if sort:
                posj = torch.argmax(probs, 1).squeeze(-1)
            elif baseline:
                posj = j
            else:
                posj = torch.multinomial(probs, 1).squeeze(-1)
            rankings.append(posj)
            if return_prob:
                log_probs[:, j] = probs[batch_index, posj].log()
            subtracts[batch_index, posj] = scores[posj] + 1e6
        rankings = torch.stack(rankings, dim=1)
        if return_prob:
            log_probs = log_probs.sum(dim=1)
            return rankings, log_probs
        else:
            return rankings

    else:
        assert scores.dim() == 2
        batch_size, candidiate_size = scores.size(0), scores.size(1)
        subtracts = scores.new_zeros((batch_size, sample_size, candidiate_size))
        batch_index = torch.arange(
            batch_size, device=scores.device).unsqueeze(1).expand(
            batch_size, sample_size)
        sample_index = torch.arange(
            sample_size, device=scores.device).expand(
            batch_size, sample_size)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(1)):
            probs = nn.functional.softmax(
                (scores.unsqueeze(1) - subtracts)/tau, dim=-1) + 1e-10
            if sort:
                posj = torch.argmax(
                    probs.reshape(batch_size * sample_size, -1),
                    1
                ).squeeze(-1).reshape(batch_size, sample_size)
            elif baseline:
                posj = torch.tensor(
                    [j] * (batch_size * sample_size)
                ).reshape(batch_size, sample_size)
            else:
                posj = torch.multinomial(
                    probs.reshape(batch_size * sample_size, -1),
                    1
                ).squeeze(-1).reshape(batch_size, sample_size)
            rankings.append(posj)
            if return_prob:
                log_probs[:, :, j] = probs[batch_index,
                                           sample_index, posj].log()
            subtracts[batch_index, sample_index,
                      posj] = scores[batch_index, posj] + 1e6
        rankings = torch.stack(rankings, dim=-1)
        if return_prob:
            log_probs = log_probs.sum(dim=-1)
            return rankings, log_probs
        else:
            return rankings

def load_searcher(path, dense=False, lexical=False, sparse=False):
    searcher = None
    if dense:
        from pyserini.search.faiss import FaissSearcher
        searcher = FaissSearcher(path, 'facebook/contriever-msmarco')
    if lexical:
        from _impact_searcher import LuceneImpactSearcher
        searcher = LuceneImpactSearcher(path, 'naver/splade-v3')
    if sparse:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(path)
        searcher.set_bm25(k1=0.9, b=0.4)
    return searcher

def get_expected_inputs(
    queries, 
    tokenizer, 
    micro_batch_inds=None
):
    tokenizer.padding_side = 'right'
    query_tensors = tokenizer(
        targets,
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).input_ids.to(query_tensors.device)
    if micro_batch_inds is not None:
        target_tensors = target_tensors[micro_batch_inds]

    input_tensors = torch.cat([query_tensors, target_tensors], -1)
    tokenizer.padding_side = 'left'

    return input_tensors, target_tensors.size(1)
