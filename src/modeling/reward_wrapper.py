import torch.nn as nn
import torch.nn.functional as F
import torch
import evaluate
from prompts.qampari import *

"""
llm = AutoModelForCausalLM.from_pretrained(
    config=config,
    low_cpu_mem_usage=train_opt.low_cpu_mem_usage,
    attn_implementation=model_opt.attn_implementation,
    stop_token_ids=stop_token_ids,
    torch_dtype=torch.bfloat16
)
reward_function = evaluate.load(evaluation_metric)

value_model = GenerativeRewardWrapper(
    generator=llm,
    tokenizer=tokenizer,
    value_name='rouge'
)
"""
class Metric:
    def __init__(self, evaluation_metric='rouge'):
        self.model = evaluate.load(evaluation_metric)

    def compute(self, xs, ys):
        results = self.model.compute(predictions=xs, references=ys, use_aggregator=False)
        results = torch.tensor(results['rouge1'])
        return results

class GenerativeRewardWrapper(nn.Module):

    def __init__(self, generator, tokenizer, reward_model):
        self.generator = generator
        self.tokenizer = tokenizer
        self.reward_model = reward_model

    def _inference(
        self, 
        queries=None,
        query_tensors=None, 
        responses=None,
        response_tensors=None,
        **generation_configs
    ):
        default = self.tokenizer.padding_side

        self.tokenizer.padding_side = 'left'
        if query_tensors is None:
            query_outputs = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.generator.device)
            query_tensors = query_outputs['input_ids']
            query_masks = query_outputs['attention_mask']
        else:
            query_masks = (query_tensors != self.tokenizer.pad_token_id)

        # 1. get response if needed
        if responses is None:
            response_outputs = self.generator.generate(
                input_ids=query_tensors,
                attention_mask=query_masks,
                **generation_configs
            )
            responses = self.tokenizer.batch_decode(
                response_outputs[:, query_tensors.shape[1]:],
                skip_special_tokens=True
            )
            ## response_outputs sometimes != response_tensors

        # 2. get response tensors # should be in the same length
        if response_tensors is None:
            self.tokenizer.padding_side = 'right'
            response_tensors = self.tokenizer(
                responses, 
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(self.generator.device)

        test = [self.tokenizer.convert_ids_to_tokens(r) for r in response_tensors]

        return query_tensors, response_tensors, responses, test

    def get_rewards(self, predictions, targets):
        results = self.reward_model.compute(
            predictions=predictions,
            references=targets,
            use_aggregator=False
        )
        results = torch.tensor(results['rouge1'])
        return results

    # @torch.no_grad()
    # def get_values(
    #     self, 
    #     query_tensors=None,
    #     target_tensors=None,
    # ):
    #     """
    #     Note that the prob is the P(wt|w<t). 
    #     So the prob would be input - 1 (the truth of the last word is meaningless.)
    #     """
    #     all_tensors = torch.cat((query_tensors, target_tensors), dim=1)
    #     label_tensors = all_tensors[..., -(target_length-1):]
    #     all_masks = all_tensors != self.tokenizer.pad_token_id
    #     position_ids = all_masks.cumsum(1) - all_masks.long()
    #     logits = self.generator(
    #         input_ids=all_tensors,
    #         attention_mask=all_masks, 
    #         position_ids=position_ids,
    #         labels=None
    #     ).logits
    #
    #     target_length = target_tensors.shape[1]
    #     query_length = query_tensors.shape[1]
    #
    #     all_logits = all_logits * all_masks.unsqueeze(-1)
    #     all_logits = all_logits[:, -target_length:, :]
    #     # logits /= self.temperature + 1e-7
    #
    #     ## get target logprob
    #     # all_logprobs = F.softmax(logits, dim=-1)
    #     all_logprobs = F.log_softmax(all_logits, dim=-1)
    #
    #     ## get target logits
    #     logits = torch.gather(all_logits, 2, label_tensors.unsqueeze(-1))
    #     logprobs = torch.gather(all_logprobs, 2, label_tensors.unsqueeze(-1))
    #     logprobs = logprobs.mean(-1)
    #
    #     del all_tensors, all_masks, all_logits, all_logprobs, label_tensors
    #     torch.cuda.empty_cache()
    #     return logits, logprobs # B |r|
    #
