import evaluate

class Metrics:

    def __init__(self, metrics):

        if isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            self.models = evalaute.load('rouge')

    def __call__(self, predictions, references):
        self.model(predictions ,references)
