import requests

config = {"max_new_tokens": 512,
          "do_sample": True,
          "num_return_sequences": 1,
          "top_k": 50,
          "top_p": 0.9,
          "temperature": 0.6,
          }

class Conversation:
    def __init__(self):
        self.history = [{"role": "system", "content": "you generate short and non offensive answers."}]

    def add_user_input(self, user_input):
        self.history.append({"role": "user", "content": user_input})

    def add_assistant_response(self, assistant_response):
        self.history.append({"role": "assistant", "content": assistant_response})

    def get_conversation(self):
        return self.history

conversation = Conversation()

while True:
    user_prompt = input("Enter your prompt, or type 'exit' to quit: ")
    if user_prompt.lower() in ['exit', 'quit']:
        break
    if user_prompt.lower() in ['history', 'h']:
        for turn in conversation.get_conversation():
            print(f"{turn['role']}: {turn['content']}")
        continue
    if user_prompt.lower() in ['clear', 'c']:
        conversation = Conversation()
        continue
    if user_prompt:
        conversation.add_user_input(user_prompt)
        sample_input = {"text": conversation.get_conversation(), "config": config, "stream": True}
        outputs = requests.post("http://127.0.0.1:8000/", json=sample_input, stream=True)
        outputs.raise_for_status()

        assistant_response = ""
        for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
            assistant_response += output
            print(output, end="", flush=True)

        conversation.add_assistant_response(assistant_response)
        print()
