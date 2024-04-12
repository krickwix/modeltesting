import requests

import concurrent.futures

# Prompt for the model
prompts = [
    "Who are the 10 most acclaimed american movie directors, and their respective 3 most acclaimed movies? Give a short plot for each movie.",
    ]

# Add generation config here
config = {
    "temperature": 0.9,
    "do_sample": True,
    "top_k": 10,
    "top_p" : 0.9,
    "num_return_sequences": 1, 
    "max_new_tokens": 2048,
    "early_stopping": True
 }

sys_snippet = "You are a helpful assistant giving straight and concise answers"
def format_prompt(prompt):
    return f"[INST]<<SYS>>{sys_snippet}<</SYS>>{prompt}[/INST]"

for prompt in prompts:
    sample_input = {"text": format_prompt(prompt), "config": config, "stream": True}
    outputs = requests.post("https://ray-llama.cantor-ai.net/", json=sample_input, stream=True)
    # outputs.raise_for_status()
    for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
        print(output, end="", flush=True)
    print()
