import asyncio
from functools import partial
from queue import Empty
from typing import Dict, Any

import torch
import logging

from ray import serve

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response, StreamingResponse
from fastapi.requests import Request
# from starlette.requests import Request

from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from transformers import TextIteratorStreamer

logger = logging.getLogger("ray.serve")

app = FastAPI()

# Define the Ray Serve deployment
# 
@serve.deployment(ray_actor_options={"num_cpus": 10, "resources": {"HPU": 1}})
class FalconModel:
    def __init__(self, model_id_or_path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        # Tweak transformers to optimize performance
        adapt_transformers_to_gaudi()

        self.device = torch.device("hpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, use_fast=False, use_auth_token=""
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        hf_config = AutoConfig.from_pretrained(
            model_id_or_path,
            torchscript=True,
            use_auth_token="",
            trust_remote_code=False,
        )
        # Load the model in Gaudi
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            config=hf_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_auth_token="",
        )
        model = model.eval().to(self.device)

        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        # Enable hpu graph runtime
        self.model = wrap_in_hpu_graph(model)

        # Set pad token, etc.
        # self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
        self.tokenizer.padding_side = "left"
        # self.tokenizer.add_eos_token = True
 
        # Use async loop in streaming
        self.loop = asyncio.get_running_loop()

    def tokenize(self, prompt: str):
        """Tokenize the input and move to HPU."""

        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = input_tokens.input_ids.to(self.device)
        attention_mask = input_tokens.attention_mask.to(self.device)
        return input_ids, attention_mask

    def generate(self, prompt: str, **config: Dict[str, Any]):
        """Take a prompt and generate a response."""

        input_ids, attention_mask = self.tokenize(prompt)
        config.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        config.setdefault("early_stopping", True)


        gen_tokens = self.model.generate(input_ids, attention_mask=attention_mask, **config)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    async def consume_streamer_async(self, streamer):
        """Consume the streamer asynchronously."""

        while True:
            try:
                for token in streamer:
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)

    def streaming_generate(self, prompt: str, streamer, **config: Dict[str, Any]):
        """Generate a streamed response given an input."""

        input_ids, attention_mask = self.tokenize(prompt)
        self.model.generate(input_ids, attention_mask=attention_mask, streamer=streamer, **config)

    async def __call__(self, http_request: Request):
        """Handle HTTP requests."""
        logger.info("Received request")
        # Load fields from the request
        json_request: str = await http_request.json()
        text = json_request["text"]
        # Config used in generation
        config = json_request.get("config", {})
        streaming_response = json_request["stream"]

        # Prepare prompts
        prompts = []
        if isinstance(text, list):
            prompts.extend(text)
        else:
            prompts.append(text)

        # Process config
        # config.setdefault("max_new_tokens", 128)

        # Enable HPU graph runtime
        config["hpu_graphs"] = True
        # Lazy mode should be True when using HPU graphs
        config["lazy_mode"] = True

        # Non-streaming case
        if not streaming_response:
            return self.generate(prompts, **config)

        # Streaming case
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True
        )
        # Convert the streamer into a generator
        self.loop.run_in_executor(
            None, partial(self.streaming_generate, prompts, streamer, **config)
        )
        return StreamingResponse(
            self.consume_streamer_async(streamer),
            status_code=200,
            media_type="text/plain",
        )

# Replace the model ID with path if necessary
entrypoint = FalconModel.bind("tiiuae/falcon-7b")