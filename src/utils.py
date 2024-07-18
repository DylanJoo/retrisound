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

# from trl.trainer.utils import first_true_indices
# def get_reward_from_policy(
#     model: torch.nn.Module, 
#     query_targets: torch.Tensor, 
#     pad_token_id: int, 
#     context_length: int
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Computes the reward logits and the rewards for a given model and query responses.
#
#     Args:
#         model (`torch.nn.Module`):
#             The model used to compute the reward logits.
#         query_targets (`torch.Tensor`):
#             The tensor containing the query targets.
#         pad_token_id (`int`):
#             The token ID representing the pad token.
#         context_length (`int`):
#             The length of the context in the query responses.
#
#     Returns:
#     """
#     attention_mask = query_targets != pad_token_id
#     position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
#     lm_backbone = getattr(model, model.base_model_prefix)
#     input_ids = torch.masked_fill(query_targets, ~attention_mask, 0)
#     output = lm_backbone(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         position_ids=position_ids,
#         return_dict=True,
#         output_hidden_states=True,
#         use_cache=False,  # otherwise mistral-based RM would error out
#     )
#     # reward_logits = model.score(output.hidden_states[-1])
#     reward_logits_of_target = model.score(output.hidden_states[-1])
#     sequence_lengths = first_true_indices(query_targets[:, context_length:] == pad_token_id) - 1 + context_length
#     # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
#     return (
#         reward_logits,
#         reward_logits[
#             torch.arange(reward_logits.size(0), device=reward_logits.device),
#             sequence_lengths,
#         ].squeeze(-1),
#         sequence_lengths,
#     )

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

