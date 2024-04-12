from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# Your existing script adapted for Flask
@app.route('/', methods=['POST', 'GET'])
def chat_with_llm():
    if request.method == 'POST':
        data = request.json
        prompt = data.get('prompt', '')
        config = {
            "temperature": 0.9,
            "do_sample": True,
            "top_k": 10,
            "top_p": 0.9,
            "num_return_sequences": 1,
            "max_new_tokens": 1536,
        }
        sys_snippet = "You are a helpful assistant giving straight and concise answers"

        def format_prompt(prompt):
            return f"[INST]<<SYS>>{sys_snippet}<</SYS>>{prompt}[/INST]"

        sample_input = {
            "text": format_prompt(prompt),
            "config": config,
            "stream": True
        }
        outputs = requests.post("https://ray-llama.cantor-ai.net/", json=sample_input, stream=True)
        # Collect all output before returning
        response_content = ""
        for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
            response_content += output
            render_template("index.html", response=response_content)
        # return jsonify({"response": response_content})
    else:
        return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True,port=8888)
