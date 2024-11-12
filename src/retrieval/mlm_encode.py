import json
import os
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer
from collections import defaultdict
import sys
from datasets import Dataset
from modeling.base_encoder import SparseEncoder

def batch_iterator(
    iterable, 
    size=1, 
    return_index=False
):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def sort_dict(
    dictionary, 
    quantization_factor,
    minimum
):
    d = {k: v*quantization_factor for (k, v) in dictionary if v >= minimum}
    sorted_d = {reverse_voc[k]: round(v, 3) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return sorted_d

def generate_vocab_vector(
    docs, 
    encoder, 
    minimum=0, 
    device='cpu', 
    max_length=512, 
    quantization_factor=100
):
    # now compute the document representation
    inputs = tokenizer(
        docs, 
        return_tensors="pt", 
        padding='max_length', 
        truncation=True, 
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        doc_reps = encoder(**inputs).reps

    # get the number of non-zero dimensions in the rep:
    cols = torch.nonzero(doc_reps)

    # now let's inspect the bow representation:
    weights = defaultdict(list)
    for col in cols:
        i, j = col.tolist()
        weights[i].append( (j, doc_reps[i, j].cpu().tolist()) )

    return [sort_dict(weight, quantization_factor, minimum) for i, weight in weights.items()]

def batch_inference(args, dataset, shard=0):
    output_dir = os.path.dirname(args.collection_output)
    os.makedirs(output_dir, exist_ok=True)

    with open(f'{args.collection_output}_{shard}', 'w') as fout:
        vectors = []
        data_iterator = batch_iterator(dataset, args.batch_size, False)

        for batch in tqdm(data_iterator, total=len(dataset)//args.batch_size+1):
            batch_vectors = generate_vocab_vector(
                    docs=batch['contents'], 
                    encoder=model,
                    minimum=args.minimum,
                    device=args.device,
                    max_length=args.max_length,
                    quantization_factor=args.quantization_factor
            )
            vectors += batch_vectors

            # collection and re-dump the collections
            n = len(batch['id'])
            for i in range(n):
                example = {
                    "id": batch['id'][i],
                    "contents": batch['contents'][i],
                    "title": batch['title'][i],
                    "vector": batch_vectors[i]
                }
                fout.write(json.dumps(example, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument('-c', "--collection", type=str)
    parser.add_argument("--collection_output", type=str)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--quantization_factor", type=int, default=1000)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--minimum", type=float, default=0)
    args = parser.parse_args()

    # load models
    model = SparseEncoder(
        model_name_or_path=args.model_name_or_path, 
        output="MLM",
        agg="max",
        activation='relu',
        norm=False
    ).eval()
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path or args.tokenizer_name)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    # load data
    with open(args.collection, 'r') as f:
        i = 0
        collection = []
        for line in tqdm(f):
            item = json.loads(line.strip())
            if 'contents' not in item.keys(): # for beir-cellar
                item['id'] = item['_id']
                item['contents'] = item['title'] + " " + item['text']
            collection.append(item)

            if len(collection) >= 1000000:
                dataset = Dataset.from_list(collection)
                print(dataset)
                batch_inference(args, dataset, i)

                i += 1
                collection = []

    # finish the rest of collections
    if len(collection) > 0:
        dataset = Dataset.from_list(collection)
        batch_inference(args, dataset, i)
