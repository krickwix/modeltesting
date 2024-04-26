from typing import Dict, Any

import ray
from ray import serve
from ray.util.queue import Queue
from ray.runtime_env import RuntimeEnv

# We need to set these variables for this example.
HABANA_ENVS = {
    "PT_HPU_LAZY_ACC_PAR_MODE": "0",
    "PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": "0",
    "PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE": "0",
    "PT_HPU_ENABLE_LAZY_COLLECTIVES": "true",
    "HABANA_VISIBLE_MODULES": "0,1,2,3,4,5,6,7",
}

class Conversation:
    def __init__(self):
        self.history = []

    def add_user_input(self, user_input):
        self.history.append({"role": "user", "content": user_input})

    def add_assistant_response(self, assistant_response):
        self.history.append({"role": "assistant", "content": assistant_response})

    def get_conversation(self):
        return self.history

import torch

@ray.remote(resources={"HPU": 1})
class DeepSpeedInferenceWorker:
    def __init__(self, model_id_or_path: str, world_size: int, local_rank: int):
        """An actor that runs a DeepSpeed inference engine.

        Arguments:
            model_id_or_path: Either a Hugging Face model ID
                or a path to a cached model.
            world_size: Total number of worker processes.
            local_rank: Rank of this worker process.
                The rank 0 worker is the head worker.
        """
        import os
        os.environ.update(HABANA_ENVS)
        os.environ["MASTER_ADDR"] = "0.0.0.0"
        from transformers import AutoTokenizer, AutoConfig
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        # Tweak transformers for better performance on Gaudi.
        adapt_transformers_to_gaudi()

        print(f"Initializing DeepSpeed worker {local_rank} with model {model_id_or_path}")
        print(f"World size: {world_size}")

        self.model_id_or_path = model_id_or_path
        self._world_size = world_size
        self._local_rank = local_rank
        self.device = torch.device("hpu")

        self.model_config = AutoConfig.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.bfloat16,
            token="",
            trust_remote_code=False,
        )

        # Load and configure the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, use_fast=True, padding_side="left"
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.eos_token = '<|eot_id|>'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        import habana_frameworks.torch.distributed.hccl as hccl

        print(f"Initializing distributed HPU on worker rank={local_rank}, world_size={world_size}, local_rank={local_rank}")
        # Initialize the distributed backend.
        hccl.initialize_distributed_hpu(
            world_size=world_size, rank=local_rank, local_rank=local_rank
        )
        torch.distributed.init_process_group(backend="hccl")
        print(f"Initialized distributed HPU on worker {local_rank}")

    def load_model(self):
        """Load the model to HPU and initialize the DeepSpeed inference engine."""
        print(f"Loading model on worker {self._local_rank}...")
        import deepspeed
        from transformers import AutoModelForCausalLM
        from optimum.habana.checkpoint_utils import (
            get_ds_injection_policy,
            write_checkpoints_json,
        )
        import tempfile

        # Construct the model with fake meta Tensors.
        # Loads the model weights from the checkpoint later.
        with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
            model = AutoModelForCausalLM.from_config(
                self.model_config, torch_dtype=torch.bfloat16,
            )
        model = model.eval()

        # Create a file to indicate where the checkpoint is.
        checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="w+")
        write_checkpoints_json(
            self.model_id_or_path, self._local_rank, checkpoints_json, token=""
        )

        # Prepare the DeepSpeed inference configuration.
        kwargs = {"dtype": torch.bfloat16}
        kwargs["checkpoint"] = checkpoints_json.name
        kwargs["tensor_parallel"] = {"tp_size": self._world_size}
        # Enable the HPU graph, similar to the cuda graph.
        kwargs["enable_cuda_graph"] = True
        # Specify the injection policy, required by DeepSpeed Tensor parallelism.
        kwargs["injection_policy"] = get_ds_injection_policy(self.model_config)

        print(f"Initializing DeepSpeed inference {self._local_rank}...")
        # Initialize the inference engine.
        self.model = deepspeed.init_inference(model, **kwargs).module

    def tokenize(self, conversation):
        """Tokenize the input and move it to HPU."""
        self.tokenizer.eos_token_id = 128009
        input_tokens = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, return_tensors="pt")
        return input_tokens.to(device=self.device)

    def generate(self, conversation, **config):
        """Take in a conversation and generate a response."""
        config["eos_token_id"] = 128009
        input_tokens = self.tokenize(conversation)
        gen_tokens = self.model.generate(input_tokens, **config)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def streaming_generate(self, conversation, streamer, **config):
        """Generate a streamed response given a conversation."""
        config["eos_token_id"] = 128009
        input_tokens = self.tokenize(conversation)
        self.model.generate(input_tokens, streamer=streamer, **config)
        
    def get_streamer(self):
        """Return a streamer.

        We only need the rank 0 worker's result.
        Other workers return a fake streamer.
        """
        
        if self._local_rank == 0:
            self.tokenizer.eos_token = "<|eot_id|>"
            return RayTextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                           skip_special_tokens=True)
        else:

            class FakeStreamer:
                def put(self, value):
                    pass

                def end(self):
                    pass

            return FakeStreamer()

from transformers import TextStreamer

class RayTextIteratorStreamer(TextStreamer):
    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        timeout: int = None,
        **decode_kwargs: Dict[str, Any],
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

# Define the Ray Serve deployment.
@serve.deployment
class DeepSpeedLlamaModel:

    def __init__(self, world_size: int, model_id_or_path: str):
        self._world_size = world_size

        # Create the DeepSpeed workers
        self.deepspeed_workers = []
        for i in range(world_size):
            print(f"Creating DeepSpeed worker {i}")
            self.deepspeed_workers.append(
                DeepSpeedInferenceWorker.options(
                    runtime_env=RuntimeEnv(env_vars=HABANA_ENVS)
                ).remote(model_id_or_path, world_size, i)
            )

        # Load the model to all workers.
        print("Loading models...")
        for worker in self.deepspeed_workers:
            worker.load_model.remote()

        # Get the workers' streamers.
        self.streamers = ray.get(
            [worker.get_streamer.remote() for worker in self.deepspeed_workers]
        )

    def generate(self, prompt: str, **config: Dict[str, Any]):
        """Send the prompt to workers for generation.

        Return after all workers finish the generation.
        Only return the rank 0 worker's result.
        """

        futures = [
            worker.generate.remote(prompt, **config)
            for worker in self.deepspeed_workers
        ]
        return ray.get(futures)[0]

    def streaming_generate(self, prompt: str, **config: Dict[str, Any]):
        """Send the prompt to workers for streaming generation.

        Only use the rank 0 worker's result.
        """

        for worker, streamer in zip(self.deepspeed_workers, self.streamers):
            worker.streaming_generate.remote(prompt, streamer, **config)

    def consume_streamer(self, streamer):
        """Consume the streamer and return a generator."""
        for token in streamer:
            yield token

    async def __call__(self, http_request: Request):
        """Handle received HTTP requests."""
        # Load fields from the request
        json_request: str = await http_request.json()
        text = json_request["text"]
        print(f"Received request: {text}")
        # Config used in generation
        config = json_request.get("config", {})
        streaming_response = json_request["stream"]

        # Prepare prompts
        prompts = []
        if isinstance(text, list):
            prompts.extend(text)
        else:
            prompts.append(text)

        # Process the configuration.
        config.setdefault("max_new_tokens", 128)

        # Enable HPU graph runtime.
        config["hpu_graphs"] = True
        # Lazy mode should be True when using HPU graphs.
        config["lazy_mode"] = True
        # Non-streaming case
        if not streaming_response:
            return self.generate(prompts, **config)

        # Streaming case
        print(f"Streaming response for {prompts}")
        self.streaming_generate(prompts, **config)
        return StreamingResponse(
            self.consume_streamer(self.streamers[0]),
            status_code=200,
            media_type="text/plain",
        )

# Replace the model ID with a path if necessary.
entrypoint = DeepSpeedLlamaModel.bind(8, "/data/models/Meta-Llama-3-70B-Instruct")
