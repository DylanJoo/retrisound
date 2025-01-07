import torch
from utils import load_searcher

s = load_searcher('/home/dju/indexes/beir-cellar/nfcorpus.lucene', lexical=True)
s.device = 'cpu'

from transformers import AutoModelForMaskedLM, AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained('naver/splade-v3')
tokenizer = AutoTokenizer.from_pretrained('naver/splade-v3')

logits = model(**tokenizer(['hello world', 'apple'], return_tensors='pt', padding=True)).logits
logits = torch.relu(logits).max(1).values

logits = torch.randint(0, 2, (1,30522))
mask = torch.rand((1, 30522)) > 0.8
logits = logits * mask

print(logits.nonzero().size(0))

# print(b)
# o = s.encode(text=None, batch_aggregated_logits=b.numpy())
# print(o)
print(logits.shape)

o = s.batch_search(
    logits=logits.detach().numpy(),
    q_ids=['1', '2'],
    k=5
)
print(o)
