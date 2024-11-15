from modeling.biencoders.sparse_new import SparseAdaptiveEncoders
from modeling.base_encoder import SparseEncoder
from options import ModelOptions
from transformers import AutoTokenizer

q_enc = SparseEncoder('naver/splade-v3', cross_attention=True)
d_enc = SparseEncoder('naver/splade-v3')

opt = ModelOptions()
bi_enc = SparseAdaptiveEncoders(opt=opt, q_encoder=q_enc, encoder=d_enc).eval()

tokenizer = AutoTokenizer.from_pretrained('naver/splade-v3')
input1 = tokenizer('NBA super star.', return_tensors='pt')
input2 = tokenizer('Stephen curry is good at shooting.', return_tensors='pt')

print(input1['input_ids'].shape)
print(input2['input_ids'].shape)

o = bi_enc(
    input1.input_ids,
    input1.attention_mask,
    input2.input_ids,
    input2.attention_mask,
    include_n_feedbacks=1
)
print(tokenizer.batch_decode(o.reps[0, -1].argsort()[-10:]))

o = bi_enc(
    input1.input_ids,
    input1.attention_mask,
)
print(tokenizer.batch_decode(o.reps[0, -1].argsort()[-10:]))
