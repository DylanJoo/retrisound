import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
import json
from tqdm import tqdm
import time
import string
import re
import yaml
from .utils import *

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

from dataclasses import dataclass
@dataclass
class default_args:
    model: str = None
    load_mode: str = 'no'
    temperature: str = 1.0
    top_p: float = 0.95

class LLM:

    def __init__(self, args):
        self.args = args

        self.model, self.tokenizer = load_model(args.model, load_mode=args.load_mode)
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0

    def generate(self, prompt, max_tokens, stop=None):
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
        stop = list(set(stop + ["<|eot_id|>", "ĊĊĊ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = [self.tokenizer.eos_token_id] + [self.tokenizer.convert_tokens_to_ids(token) for token in stop]
        stop_token_ids = list(set([token_id for token_id in stop_token_ids if token_id is not None]))
        logger.warning("Terminration token ids: " + ', '.join( [str(id) for id in stop_token_ids] ))
        logger.warning("Terminration tokens: " + ', '.join(self.tokenizer.convert_ids_to_tokens(stop_token_ids)))
        outputs = self.model.generate(
            **inputs,
            do_sample=True, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            eos_token_id=stop_token_ids
        )
        generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        generation = self.postprocess(generation)
        return generation

    def postprocess(self, x):
        x = x.split('\n\n')[0] 
        x = x.split('Question: ')[0] 
        return x
