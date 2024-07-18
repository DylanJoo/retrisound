from trl.trainer.utils import *
import torch

def update_tokenizer(
    tokenizer, 
    pad_token='<|reserved_special_token_0|>', 
):
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

# add max length constraints
def get_expected_inputs(query_tensors, targets, tokenizer):
    tokenizer.padding_side = 'right'
    target_tensors = tokenizer(
        targets,
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).input_ids.to(query_tensors.device)
    input_tensors = torch.cat([query_tensors, target_tensors], -1)
    tokenizer.padding_side = 'left'
    return input_tensors, target_tensors.size(1)
