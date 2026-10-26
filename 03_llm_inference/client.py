"""Ask the vLLM server running in your Slurm job (serve_vllm.sh) a question.

    source vllm-server.env           # written by serve_vllm.sh in this folder
    python3 client.py "What is a job scheduler?"

Standard library only, so it runs on the login node without any environment.
The server speaks the OpenAI API, so the `openai` package, LangChain, etc. work too.
"""
import json
import os
import sys
import urllib.request

try:
    base_url, api_key, model = (os.environ[k] for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "VLLM_MODEL"))
except KeyError:
    sys.exit("Connection details not set: source vllm-server.env first")

question = " ".join(sys.argv[1:]) or "Hello! Who are you?"
body = {
    "model": model,
    "messages": [{"role": "user", "content": question}],
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
with urllib.request.urlopen(request, timeout=300) as response:
    answer = json.load(response)
print(answer["choices"][0]["message"]["content"])
print(f"\n[{answer['usage']['completion_tokens']} tokens]")
