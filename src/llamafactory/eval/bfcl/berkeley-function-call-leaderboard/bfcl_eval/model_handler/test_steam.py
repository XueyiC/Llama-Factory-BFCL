import requests
import json

# 测试流式响应的完整内容
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen3-4B-Thinking-2507",
        "messages": [{"role": "user", "content": "What is 2+2? Think step by step."}],
        "stream": True,
        "extra_body": {"enable_thinking": True},
        "temperature": 0.7
    },
    stream=True
)

print("=== 完整的流式响应 ===\n")
full_content = ""
chunk_count = 0

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith("data: "):
            chunk_count += 1
            data_str = line_str[6:]  # 移除 "data: " 前缀
            
            if data_str == "[DONE]":
                print(f"\n=== 流结束 ===")
                break
                
            try:
                chunk = json.loads(data_str)
                
                # 打印每个 chunk 的结构（前5个）
                if chunk_count <= 5:
                    print(f"Chunk {chunk_count}: {json.dumps(chunk, indent=2)}")
                
                # 收集内容
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    
                    # 检查各种可能的字段
                    if "content" in delta:
                        full_content += delta["content"]
                    
                    # 检查是否有其他字段
                    if chunk_count == 1:  # 只在第一个 chunk 打印
                        print(f"\nDelta 字段: {delta.keys()}\n")
                        
            except json.JSONDecodeError:
                print(f"解析错误: {data_str}")

print(f"\n=== 完整内容 ===")
print(full_content)

# 检查是否有 <think> 标签
if "<think>" in full_content:
    print("\n✅ 发现 <think> 标签在 content 中")
else:
    print("\n❌ 没有发现 <think> 标签")