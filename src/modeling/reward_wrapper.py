import re
import torch.nn as nn
import torch.nn.functional as F
import torch
import evaluate
from prompts.qampari import *
from torch.nn import CrossEntropyLoss

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

    def compute(self, **kwargs):
        results = self.model.compute(**kwargs)
        results = torch.tensor(results['rouge1'])
        return results

class Judgement:
    def __init__(self, scales=None):
        if scales:
            self.scale = scales
        else:
            self.scale = list(range(0, 6)) # 0, 1, ...., 5

    def compute(self, judgements):
        pattern = re.compile(r"[\d\.\d]+")

        results = [0] * len(judgements)
        for i, judgement in enumerate(judgements):
            result = re.findall(pattern, judgement + "-1")[0]
            try:
                result = min(self.scale) if len(result) == 0 else float(result)
                results[i] = min(result, self.scale[-1])
            except:
                results[i] = float(0)
        results = torch.tensor(results).float()
        return results

class GenerativeRewardWrapper(nn.Module):

    def __init__(self, generator, tokenizer, utility, generation_config):
        super().__init__()
        self.generator = generator
        self.tokenizer = tokenizer
        self.utility = utility
        self.generator.generation_config = generation_config
        self.generator.generation_config.pad_token_id = tokenizer.pad_token_id

        # freeze params
        for n, p in self.generator.named_parameters():
            p.requires_grad = False

    def _inference(
        self, 
        queries=None,
        query_tensors=None,
        max_new_tokens=64
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
        response_outputs = self.generator.generate(
            input_ids=query_tensors,
            attention_mask=query_masks,
            do_sample=True, 
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
        )
        responses = self.tokenizer.batch_decode(
            response_outputs[:, query_tensors.shape[1]:],
            skip_special_tokens=True
        )
        responses = [self.normalize(r) for r in responses]

        # 2. get response tensors # should be in the same length
        self.tokenizer.padding_side = 'right'
        response_tensors = self.tokenizer(
            responses, 
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(self.generator.device)

        # test = [self.tokenizer.convert_ids_to_tokens(r) for r in response_tensors]
        # return query_tensors, response_tensors, responses, test
        return query_tensors, response_tensors, responses

    def get_rewards(self, predictions, targets=None):
        if isinstance(self.utility, Metric):
            results = self.utility.compute(
                predictions=predictions,
                references=targets,
                use_aggregator=False
            )

        if isinstance(self.utility, Judgement):
            results = self.utility.compute(judgements=predictions)

        return results

    @staticmethod
    def normalize(texts):
        texts = texts.strip()
        pattern = re.compile(r"\s+")
        texts = re.sub(pattern, ' ', texts).strip()
        pattern = re.compile(r"\n")
        texts = re.sub(pattern, ' ', texts).strip()
        ## remove q tag
        texts = texts.split('</q>')[0]
        return texts

    @torch.no_grad()
    def get_likelihood(
        self, 
        queries=None,
        query_tensors=None,
        targets=None,
        max_length=2048,
    ):
        device = self.generator.device

        self.tokenizer.padding_side = 'right'
        target_tensors = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(device)
        max_length = self.generator.config.max_position_embeddings 
        target_length = target_tensors.size(1)
        max_query_length = max_length - target_length

        if query_tensors is None:
            self.tokenizer.padding_side = 'left'
            query_tensors = self.tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_query_length
            )['input_ids'].to(device)

        query_tensors = query_tensors[:, :max_query_length]

        # craft inputs
        input_ids = torch.cat( [query_tensors, target_tensors], -1)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(device)
        position_ids = attention_mask.cumsum(1) - attention_mask.long()  

        logits = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None
        ).logits

        # target_logits = logits[:, -target_length:-1, :]
        target_logits = logits[:, -target_length:, :]
        label_tensors = input_ids.clone()
        label_tensors[label_tensors == self.tokenizer.pad_token_id] = -100
        label_tensors = label_tensors[..., -target_length:]
        del input_ids
        torch.cuda.empty_cache()

        target_lengths = [len(mask!=0) for mask in attention_mask[..., -target_length:]]
        neg_likelihood = self.compute_nll(target_logits, label_tensors, target_lengths)
        return -neg_likelihood

    def compute_nll(self, logits, labels, lengths=None):
        ## extract the batch-wise mean
        batch_size, _, vocab_size = logits.shape
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        nll = loss_fct(shift_logits, shift_labels).view(batch_size, -1)

        nll_to_return = []
        if lengths is not None:
            for i, length in enumerate(lengths):
                nll_to_return.append(nll[i, :length].mean())
            nll_to_return = torch.tensor(nll_to_return).flatten()
        else:
            nll_to_return = nll.view(batch_size, -1).mean(-1)

        return nll_to_return.flatten()

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

    # def inference(
    #     self, 
    #     queries=None,
    #     query_tensors=None, 
    #     batch_size=None
    # ):
    #     if batch_size is None:
    #         return self._inference(queries, query_tensors, responses)
    #     else:
    #         query_tensors, response_tensors, responses = [], [], []
    #         for i in range(0, queries.shape[0], batch_size):
    #             b_queries = queries[i: i+batch_size]
    #             if query_tensors is not None:
    #                 qt, rt, r = self._inference(
    #                     query_tensors=query_tensors[i: i+batch_size],
    #                 )
    #             else:
    #                 qt, rt, r = self._inference(
    #                     queries=queries[i: i+batch_size],
    #                 )
    #             query_tensors.append(qt)
    #             response_tensors.append(rt)
    #             responses.append(r)
    #         query_tensors = torch.cat(query_tensors, 0)
    #         response_tensors = torch.cat(response_tensors, 0)
    #         responses += r
    #         return query_tensors, response_tensors, responses
