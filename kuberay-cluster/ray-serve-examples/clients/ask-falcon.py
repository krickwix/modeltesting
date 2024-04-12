import requests

import concurrent.futures
# Prompt for the model
prompts = [
    "Tell me a joke about falcon"
    # "Who are the 10 most acclaimed american movie directors?",
    ]

# Add generation config here
config = {
    "temperature": 0.9,
    "do_sample": True,
    "top_k": 10,
    "top_p" : 0.9,
    "num_return_sequences": 1, 
    "max_new_tokens": 200,
    # "early_stopping": True
    "num_beams": 1,
    "max_length": 200,
 }

sys_snippet = "You are a helpful assistant giving straight and concise answers"
def format_prompt(prompt):
    return f"[INST]{prompt}[/INST]"

for prompt in prompts:
    sample_input = {"text": format_prompt(prompt), "config": config, "stream": True}
    outputs = requests.post("https://ray-falcon.cantor-ai.net/", json=sample_input, stream=True)
    # outputs.raise_for_status()
    for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
        print(output, end="", flush=True)
    print()
