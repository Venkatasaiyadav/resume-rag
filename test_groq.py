"""Quick test to verify Groq is working"""

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    messages=[
        {"role": "user", "content": "Say 'Hello, RAG is working!' in a fun way."}
    ],
    max_tokens=100,
)

print("✅ Groq Response:", response.choices[0].message.content)
print("📊 Tokens used:", response.usage.total_tokens)