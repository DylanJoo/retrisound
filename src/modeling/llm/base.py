import argparse
import os
import json
from tqdm import tqdm
import time
import string
import re
import random
from nltk import sent_tokenize

from vllm import LLM as v_llm 
from vllm import SamplingParams
from transformers import AutoTokenizer


class vLLM:

    def __init__(self, model, temperature=0.7, top_p=0.9, num_gpus=1, gpu_memory_utilization=0.75):
        self.model = v_llm(
            model, 
            dtype='half',
            enforce_eager=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False
        )

    def generate(self, x, **kwargs):
        self.sampling_params.max_tokens = kwargs.pop('max_tokens', 256)
        self.sampling_params.min_tokens = kwargs.pop('min_tokens', 32)
        outputs = self.model.generate(x, self.sampling_params, use_tqdm=False)
        if len(outputs) > 1:
            return [o.outputs[0].text for o in outputs]
        else:
            return [outputs[0].outputs[0].text]

class dummyLLM:

    def generate(self, inputs):
        if len(inputs[0]) > 512:
            outputs = []
            for input in inputs:
                strings = input.split()
                random.shuffle(strings)
                outputs.append( " ".join(strings[:100]) )
            return outputs
        else:
            return inputs

class LLM:

    def __init__(self, args):
        self.args = args

        self.model, self.tokenizer = load_model(
            args.model, 
            load_mode=args.load_mode, 
            flash_attention_2=args.ampere_gpu
        )
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0

    def generate(self, prompt, max_tokens, min_tokens=None, stop=None):
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        stop = [] if stop is None else stop
        # ['h', 'ellow', 'or', 'l', 'ĊĊ']
        # ['h', 'ellow', 'or', 'l', 'ĊĊĊ']
        stop = list(set(stop + ["<|eot_id|>", "ĊĊĊ", "ĊĊ", "<0x0A>", "```"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = [self.tokenizer.eos_token_id] + [self.tokenizer.convert_tokens_to_ids(token) for token in stop]
        stop_token_ids = list(set([token_id for token_id in stop_token_ids if token_id is not None]))
        logger.warning("Terminration token ids: " + ', '.join( [str(id) for id in stop_token_ids] ))
        logger.warning("Terminration tokens: " + ', '.join(self.tokenizer.convert_ids_to_tokens(stop_token_ids)))
        outputs = self.model.generate(
            **inputs,
            do_sample=True, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            min_new_tokens=min_tokens,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            eos_token_id=stop_token_ids
        )
        generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache()
        return generation

