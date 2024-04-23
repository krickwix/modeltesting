import requests

# Prompt for the model
prompt = "Describe quantum physics in one short sentence of no more than 12 words"

# prompt = "<|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|>Hi!"
# prompt = "Hi!"
# Add generation config here
config = {"max_new_tokens": 1024,
          "do_sample": True,
          "num_return_sequences": 1,
          "top_k": 50,
          "top_p": 0.01,
          "temperature": 0.01}

# # Non-streaming response
sample_input = {"text": prompt, "config": config, "stream": False}
# outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=False)
# print(outputs.text, flush=True)

# Streaming response
sample_input["stream"] = True
outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=True)
outputs.raise_for_status()
for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
    print(output, end="", flush=True)
print()
