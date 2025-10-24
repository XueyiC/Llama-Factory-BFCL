import requests
import json

# 直接调用 LlamaFactory API
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen3-4B-Thinking-2507",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.7
    }
)

data = response.json()
print(json.dumps(data, indent=2))

# 检查 reasoning_content 的位置
if "choices" in data and len(data["choices"]) > 0:
    message = data["choices"][0]["message"]
    if "reasoning_content" in message:
        print("\n✅ Found reasoning_content in message!")
    else:
        print("\n❌ No reasoning_content in message")
        print("Available fields:", message.keys())