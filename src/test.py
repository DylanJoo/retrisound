import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from options import ModelOptions, TrainOptions
from memory_profiler import profile

@profile
def main():
    ## prepare kwargs
    R_model_name_or_path='facebook/contriever'
    # G_model_name_or_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    G_model_name_or_path='meta-llama/Meta-Llama-3-8B-Instruct'
    model_opt = ModelOptions(
            retriever_name_or_path=R_model_name_or_path,
            generator_name_or_path=G_model_name_or_path,
    )
    train_opt = TrainOptions()

    ## prepare bi-encoders
    tokenizer_r = AutoTokenizer.from_pretrained(R_model_name_or_path)
    from modeling.rmt import RMTEncoder
    from modeling.rife import Contriever
    model = Contriever.from_pretrained(R_model_name_or_path)
    encoder = deepcopy(model)
    ada_encoder = RMTEncoder(
            base_model=model, 
            num_mem_tokens=4,
            tokenizer=tokenizer_r,
            input_size=512,
            sum_loss=False
    )
    from modeling import InBatchInteraction
    bi_encoders = InBatchInteraction(
            model_opt, 
            q_encoder=ada_encoder,
            d_encoder=encoder,
            fixed_d_encoder=True
    )

    ## prepare generator
    tokenizer_g = AutoTokenizer.from_pretrained(G_model_name_or_path, use_fast=False)

    ### [FIX] missing pad token
    ### Regarding `forward` passing, llama3 include the pad token to achieve batch forward. However, the original pre-traing did not use pad. We here add a pseudo pad token (and you should ignore the loss from the MLE of them)
    if tokenizer_g.pad_token is None:
        tokenizer_g.add_special_tokens(
            {'pad_token': '<|reserved_special_token_0|>'}
        )

    stop = ["<|eot_id|>", "ĊĊĊ", "ĊĊ", "<0x0A>"]
    stop_token_ids = [tokenizer_g.eos_token_id] + [tokenizer_g.convert_tokens_to_ids(token) for token in stop]
    stop_token_ids = list(set([token_id for token_id in stop_token_ids if token_id is not None]))
    # tokenizer_g.eos_token_id = stop_token_ids
    generator = AutoModelForCausalLM.from_pretrained(
        G_model_name_or_path,
        low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
        pad_token_id=tokenizer_g.eos_token_id
    )

    ### Resize embeds as long as pad token is out-of-vocab.
    # if len(tokenizer_g) > generator.get_input_embeddings().weight.shape[0]:
    #     model.resize_token_embeddings(len(tokenizer_g))

    from modeling import RerankAugmentedGeneration
    model = RerankAugmentedGeneration(
        llm=generator, 
        tokenizer=tokenizer_g,
        biencoders=bi_encoders,
        stop_token_ids=stop_token_ids
    )

    ## add data
    split='test'
    from data.qampari import ContextQADataset
    dataset = ContextQADataset(
        data_file=f'/home/dju/datasets/qampari/{split}_data.jsonl',
        n_max_segments=10,
        n_max_candidates=50,
        budget=5,
        depth=10,
        corpus_file='/home/dju/datasets/qampari/wikipedia_chunks/chunks_v5',
        retrieval_file=f'/home/dju/datasets/qampari/{split}_data_bm25-top100.run',
        quick_test=True
    )
    dataset.add_action(0, 'this is a testing action')

    features = [dataset[i] for i in range(4)]
    from data.qampari import ContextQACollator
    collator = ContextQACollator(
        tokenizer_r=tokenizer_r,
        tokenizer_g=tokenizer_g
    )
    d=collator(features)
    # print(d['inputs_for_retriever'])
    o=model(**d)
    # print(o.keys())
    # print(o.loss_g)

    print(o.prompts_fbk[0])
    for i, fbk in enumerate(o.feedbacks):
        print('q,', dataset[i]['question'])
        print('a,', dataset[i]['answers'])
        print('fbk,', fbk)

if __name__ == '__main__':
    main()
