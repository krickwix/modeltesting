## python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-70b-chat-hf \
## --engine-use-ray --device hpu --tensor-parallel-size 8 --worker-use-ray --max-num-seqs 4


from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

class Conversation:
    def __init__(self):
        self.history = [{"role": "system", "content": "you are a helpful assistant."}]

    def add_user_input(self, user_input):
        self.history.append({"role": "user", "content": user_input})

    def add_assistant_response(self, assistant_response):
        self.history.append({"role": "assistant", "content": assistant_response})

    def get_conversation(self):
        return self.history

conversation = Conversation()

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

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
        assistant_response = client.chat.completions.create(
                                                    model="meta-llama/Meta-Llama-3-70B-Instruct",
                                                    messages=conversation.get_conversation(),stream=True)
        result = ""
        for chunk in assistant_response:
            if chunk.choices[0].delta.content:
              result += chunk.choices[0].delta.content
              print(chunk.choices[0].delta.content,end="",flush=True)
        
        conversation.add_assistant_response(result)
        print("\n**")
