# from transformers import LlamaTokenizer, LlamaTokenizerFast
# def init_tokenizer(tokenizer, use_special_tokens):
#     # no default pad token for llama!
#     # here we add all special tokens again, because the default ones are not in the special_tokens_map
#     if use_special_tokens is True:
#         # special_token_dict = {"additional_special_tokens": ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]}
#         special_token_dict = {"additional_special_tokens": [f"[{i}]" for i in range(10)] }
#
#     special_token_dict["bos_token"] = "<s>"
#     special_token_dict["eos_token"] = "</s>"
#     special_token_dict["unk_token"] = "<unk>"
#     special_token_dict["pad_token"] = "<pad>"
#     num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
#     
#     context_markups = []
#     for token in ["<paragraph>", "</paragraph>"]:
#         context_markups.append(tokenizer.convert_tokens_to_ids(token))
#     if use_special_tokens is False:
#         assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
#     else:
#         assert num_added_tokens > 10, "special tokens must be added to the original tokenizers."
#
#     return tokenizer, context_markups

def update_tokenizer(
    tokenizer, 
    pad_token='<|reserved_special_token_0|>', 
    add_additional_tokens=None
):
    """
    pad_token [str]: default is the preserved token of llama3. 
        This should be re-considered if trying to fine-tune.
    add_special_tokens [dict]: optional additional tokens.
    """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': pad_token})
    if add_additional_tokens:
        pass

    return tokenizer
