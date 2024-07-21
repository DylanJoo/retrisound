import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-4}GB' # original is -6
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def load_model(
    model_name_or_path, 
    dtype=torch.float16, 
    load_mode=None, 
    reserve_memory=10, 
    flash_attention_2=False
):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # load_mode: whether to use int8/int4 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    logger.warn(f"Use LLM.{load_mode}")
    start_time = time.time()

    if flash_attention_2:
        model_kwargs = {
            'torch_dtype': torch.bfloat16,
            'attn_implementation': "flash_attention_2"
        }
    else:
        model_kwargs = {'torch_dtype': dtype}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        load_in_8bit=(load_mode == '8bit'),
        load_in_4bit=(load_mode == '4bit'),
        **model_kwargs
    )
        # max_memory=get_max_memory(),
        # attn_implementation="sdpa",
    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer
