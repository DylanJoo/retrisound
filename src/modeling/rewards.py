import torch
import evaluate

class MetricRewards:

    def __init__(self, evaluation_metric='rouge'):
        self.model = evaluate.load(evaluation_metric)

    def calculate_rewards(self, xs, ys):
        results = self.model.compute(predictions=xs, references=ys, use_aggregator=False)
        results = torch.tensor(results['rouge1'])
        return results

