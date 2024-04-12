# import requests

# import concurrent.futures

# # Prompt for the model
# prompts = [
#     "Who are the 10 most acclaimed american movie directors, and their respective 3 most acclaimed movies? Give a short plot for each movie.",
#     ]

# # Add generation config here
# config = {
#     "temperature": 0.9,
#     "do_sample": True,
#     "top_k": 10,
#     "top_p" : 0.9,
#     "num_return_sequences": 1, 
#     "max_new_tokens": 1024,
#     "early_stopping": True
#  }

# sys_snippet = "You are a helpful assistant giving straight and concise answers"
# def format_prompt(prompt):
#     return f"[INST]<<SYS>>{sys_snippet}<</SYS>>{prompt}[/INST]"

# for prompt in prompts:
#     sample_input = {"inputs": prompt, "parameters": config, "stream": True}
#     outputs = requests.post("http://a1968708b897748229256653f64b5caa-1426232770.us-east-1.elb.amazonaws.com/", json=sample_input, stream=True)
#     # outputs.raise_for_status()
#     for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
#         print(output, end="", flush=True)
#     print()

# from huggingface_hub import InferenceClient
# try:
#     client = InferenceClient("http://a1968708b897748229256653f64b5caa-1426232770.us-east-1.elb.amazonaws.com/")
#     response = client.text_generation("<s>[INST]Write a code for snake game[/INST]</s>")
#     print(response)
# except Exception as e:
#     print("An error occurred:", e)

# client.text_generation(prompt="Write a code for snake game")
# for token in client.text_generation("How do you make cheese?", stream=False):
#     print(token)


import requests

# Replace with your actual deployment URL
url = "http://a1968708b897748229256653f64b5caa-1426232770.us-east-1.elb.amazonaws.com/generate"

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-hf")
inputs = tokenizer("Write a code for snake game", return_tensors="pt")
# Example request data and headers
data = {"inputs": inputs, "parameters": {"temperature": 0.9, "do_sample": True, "top_k": 10, "top_p": 0.9, "num_return_sequences": 1, "max_new_tokens": 1024, "early_stopping": True}}
headers = {'Content-Type': 'application/json'}

# Send a POST request
response = requests.post(url, json=data, headers=headers)

# Check the response
print("Status Code:", response.status_code)
print("Response Body:", response.text)

