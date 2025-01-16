# from transformers import AutoModelForMaskedLM
# model = AutoModelForMaskedLM.from_pretrained('naver/splade-v3')

from modeling.base_encoder import SparseEncoder
# model = SparseEncoder('naver/splade-v3')
model = SparseEncoder(model_name_or_path='naver/splade-v3', cross_attention=False)

for n, p in self.named_parameters():
    if 'cross_encoder' in n:
        p.requires_grad = True
    elif 'transform' in n:
        p.requires_grad = True
    else:
        p.requires_grad = False

from peft import LoraConfig
peft_config = LoraConfig(
    r=512,
    lora_alpha=1024,
    init_lora_weights="gaussian",
    target_modules=["decoder"],
)
model.model.add_adapter(peft_config, adapter_name='lm')

print(sum(p.numel() for p in model.parameters() if p.requires_grad))
