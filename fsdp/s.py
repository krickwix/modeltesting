import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import habana_frameworks.torch.distributed.hccl
from torch.optim import SGD
import torch.distributed as dist
import torch.multiprocessing as mp
import os

os.environ["PT_HPU_LAZY_MODE"] = "0"

device_hpu = torch.device('hpu')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '172.17.0.2'
    os.environ['MASTER_PORT'] = '12355'
    import habana_frameworks.torch.distributed.hccl
    print(f"setup rank={rank}, world_size={world_size}")
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def test(rank, world_size):
    setup(rank, world_size)
    print(f"Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "/data/models/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(device_hpu)
    tokenizer = AutoTokenizer.from_pretrained("/data/models/Llama-2-7b-chat-hf", trust_remote_code=True)
    model = FSDP(base_model, device_id = device_hpu)

if __name__ == "__main__":
    world_size = 8
    mp.spawn(test, args=(world_size,), nprocs=world_size, join=True)