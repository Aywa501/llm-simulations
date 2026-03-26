"""Quick API sanity check — run this to verify key, model, and token usage."""
import os
from openai import OpenAI

client = OpenAI()

print(f"API key prefix: {os.environ.get('OPENAI_API_KEY', 'NOT SET')[:12]}...")

response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Reply with the single word: hello"}],
    max_completion_tokens=10,
)

print(f"Response     : {response.choices[0].message.content}")
print(f"Model used   : {response.model}")
print(f"Tokens used  : prompt={response.usage.prompt_tokens} "
      f"completion={response.usage.completion_tokens} "
      f"total={response.usage.total_tokens}")
print(f"System fingerprint: {response.system_fingerprint}")
