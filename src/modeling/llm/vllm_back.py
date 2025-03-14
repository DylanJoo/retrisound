import vllm
from typing import List

class LLM:

    def __init__(self, 
        model, 
        temperature=0.7, top_p=1.0, 
        dtype='half', gpu_memory_utilization=0.75, 
        num_gpus=1, 
        think_activated=False,
        **kwargs
    ):
        self.model = vllm.LLM(
            model, 
            dtype=dtype,
            enforce_eager=True,
            tensor_parallel_size=num_gpus,
            max_num_batched_tokens=kwargs.get('max_num_batched_tokens', 20480),
            max_model_len=kwargs.get('max_model_len', 20480),
            enable_chunked_prefill=True,
            gpu_memory_utilization=gpu_memory_utilization
        )
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False
        )
        self.think_activated = think_activated

    def preprocess(self, inputs: List):
        if self.think_activated:
            inputs = [i + "<think>\n" for i in inputs]
        return inputs

    def postprocess(self, outputs: List):

        outputs_think = [None for _ in range(len(outputs))]

        if self.think_activated:
            for i, o in enumerate(outputs):
                if '</think>' in o:
                    splits = o.split('</think>')
                    # outputs_think[i] = splits[0]
                    outputs[i] = splits[-1]

        return outputs

    def generate(self, x, max_tokens=256, min_tokens=0, **kwargs):
        self.sampling_params.max_tokens = max_tokens
        self.sampling_params.min_tokens = min_tokens

        if isinstance(x, str):
            x = [x]

        x = self.preprocess(x)
        outputs = self.model.generate(x, self.sampling_params, use_tqdm=False)

        if len(outputs) > 1:
            return self.postprocess([o.outputs[0].text for o in outputs])
        else:
            return self.postprocess([outputs[0].outputs[0].text])

