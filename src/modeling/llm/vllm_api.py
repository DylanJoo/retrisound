import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
import uuid

# cleaning
import contextlib
import gc
import torch
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel
)

def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()

class LLM:

    def __init__(self, 
        model, 
        temperature=0.7, top_p=1.0, 
        dtype='half', gpu_memory_utilization=0.75, 
        num_gpus=1, 
        enforce_eager=False,
        think_activated=False,
        **kwargs
    ):
        args = AsyncEngineArgs(
            model=model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=kwargs.pop('max_model_len', 10240)
        )
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            skip_special_tokens=False
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def _iterate_over_output(self, output_iterator: AsyncStream) -> str:
        """Process the streaming output and return the final result"""
        last_text = ""
        async for output in output_iterator:
            last_text = output.outputs[0].text
        return last_text
    
    async def _generate_async(self, prompts, sampling_params):
        """Handle the async generation process"""
        # Create unique request IDs
        request_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        
        # Add requests to the engine
        output_iterators = [
            await self.engine.add_request(request_id, prompt, sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]
        
        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator)
            for output_iterator in output_iterators
        ])
        
        return list(outputs)
    
    def generate(self, prompts, **kwargs):
        """Generate completions for the provided prompts"""
        # If prompts is a single string, convert to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Create sampling params with updated values
        sampling_params = self.sampling_params
        sampling_params.max_tokens = kwargs.pop('max_tokens', 256)
        sampling_params.min_tokens = kwargs.pop('min_tokens', 0)

        return self.loop.run_until_complete(self._generate_async(prompts, sampling_params))
