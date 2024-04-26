import requests

config = {"max_new_tokens": 4096,
          "do_sample": True,
          "num_return_sequences": 1,
          "top_k": 50,
          "top_p": 0.9,
          "temperature": 0.6}

while True:
    user_prompt = input("Enter your prompt, or type 'exit' to quit: ")
    if user_prompt.lower() in ['exit', 'quit']:
        break
    if user_prompt:
        prompt = user_prompt
    sample_input = {"text": prompt, "config": config, "stream": True}
    outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=True)
    outputs.raise_for_status()
    for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
        print(output, end="", flush=True)
    print()
