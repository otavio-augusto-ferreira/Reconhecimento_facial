import os
import sys

from openai import OpenAI

BASE_URL = "https://llm.liaufms.org/v1/qwen2-5-14b-instruct-awq"
MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"

token = os.environ.get("LLM_API_KEY")
if not token:
    sys.exit("Defina a variável de ambiente LLM_API_KEY antes de rodar.")

client = OpenAI(base_url=BASE_URL, api_key=token)
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Responda apenas: ok"}],
)
print(resp.choices[0].message.content)
