from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# 测试流式响应
stream = client.chat.completions.create(
    model="Qwen3-4B-Thinking-2507",
    messages=[{"role": "user", "content": "What is 2+2? Think step by step."}],
    stream=True,
    extra_body={"enable_thinking": True}
)

print("=== 使用 OpenAI Client ===\n")
full_content = ""
reasoning_content = ""

for chunk in stream:
    if chunk.choices[0].delta:
        delta = chunk.choices[0].delta
        
        # 打印 delta 的所有属性（只打印第一次）
        if not full_content and not reasoning_content:
            print(f"Delta attributes: {dir(delta)}\n")
            for attr in dir(delta):
                if not attr.startswith('_'):
                    try:
                        value = getattr(delta, attr)
                        if not callable(value):
                            print(f"  {attr}: {value}")
                    except:
                        pass
            print("\n---\n")
        
        # 收集内容
        if hasattr(delta, "content") and delta.content:
            full_content += delta.content
            
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content += delta.reasoning_content

print(f"Content: {full_content[:200]}...")
if reasoning_content:
    print(f"Reasoning: {reasoning_content[:200]}...")
else:
    print("No reasoning_content field found")

# 检查 content 中是否包含 thinking
if "<think>" in full_content:
    import re
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, full_content, re.DOTALL)
    if match:
        print(f"\nExtracted reasoning from <think> tags:")
        print(match.group(1)[:200] + "...")