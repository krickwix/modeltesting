from flask import Flask, request, jsonify, render_template, Response
from openai import OpenAI
from flask_sse import sse
import time

app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost"
app.register_blueprint(sse, url_prefix='/stream')

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

class Conversation:
    def __init__(self):
        self.history = []

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_prompt = request.json.get("prompt")
    global conversation
    if user_prompt.lower() in ['exit', 'quit']:
        return jsonify({"message": "Goodbye!"}), 200
    if user_prompt.lower() in ['history', 'h']:
        history = conversation.get_conversation()
        return jsonify({"history": history}), 200
    if user_prompt.lower() in ['clear', 'c']:
        conversation = Conversation()
        return jsonify({"message": "Conversation cleared!"}), 200
    if user_prompt:
        conversation.add_user_input(user_prompt)
        assistant_response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=conversation.get_conversation(), stream=True
        )

        def generate():
            result = ""
            for chunk in assistant_response:
                if chunk.choices[0].delta.content:
                    data = chunk.choices[0].delta.content
                    result += data
                    sse.publish({"message": data}, type='response')
            conversation.add_assistant_response(result)

        return Response(generate(), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
