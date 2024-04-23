# RUN : PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=0 PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE=0 PT_HPU_ENABLE_LAZY_COLLECTIVES=true deepspeed --no_local_rank --num_gpus 8 --num_nodes 1  --bind_cores_to_rank ds.py

import tempfile
import deepspeed
import torch
import transformers
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
from optimum.habana.checkpoint_utils import model_on_meta, get_ds_injection_policy, write_checkpoints_json
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import pipeline, set_seed

user_input = [
    "Who are the 10 most acclaimed movie directors of all time, and their respective 3 most acclaimed movies? Provide a structures answer using bullet points.",
]

max_new_tokens = 2048
dtype = torch.bfloat16

set_seed(42)

world_size, rank, local_rank = initialize_distributed_hpu()
import habana_frameworks.torch.distributed.hccl as hccl
hccl.initialize_distributed_hpu(
            world_size=world_size, rank=local_rank, local_rank=local_rank)
deepspeed.init_distributed(dist_backend="hccl",distributed_port=29975)

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

model_id = "/models/Meta-Llama-3-70B"

model_config = AutoConfig.from_pretrained(model_id,
                                          torch_dtype=dtype,
                                          torchscript=True,
                                          trust_remote_code=False)

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
    model = AutoModelForCausalLM.from_config(
        model_config, torch_dtype=torch.bfloat16
    )
model = model.eval()

# Create a file to indicate where the checkpoint is.
checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="w+")
write_checkpoints_json(
    model_id, local_rank, checkpoints_json
)

model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer))

ds_inference_kwargs = {"dtype": dtype}
ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
ds_inference_kwargs["enable_cuda_graph"] = True
ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(model_config)
ds_inference_kwargs["checkpoint"] = checkpoints_json.name

# print("Initializing DeepSpeed...")
ds_engine = deepspeed.init_inference(model=model,
                                    **ds_inference_kwargs)
model = ds_engine.module

for i in user_input:
    generate_ids = model.generate(input_ids=tokenizer.encode(i, return_tensors="pt",padding=True).to(model.device),
                                max_length=2048,
                                num_return_sequences=1, do_sample=True, 
                                temperature=0.9, top_k=50, top_p=0.95, 
                                pad_token_id=tokenizer.eos_token_id, 
                                eos_token_id=tokenizer.eos_token_id, 
                                bos_token_id=tokenizer.bos_token_id, 
                                use_cache=True,  
                                return_dict_in_generate=False)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print()
        print(response)
        print()
