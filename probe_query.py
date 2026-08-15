import requests
import json
import base64
import sys
from pathlib import Path

BASE = "http://localhost:8000"

# 1. 健康检查
r = requests.get(f"{BASE}/health", timeout=10)
print("1. /health:", r.status_code, r.text[:200])

# 2. 文档列表
r = requests.get(f"{BASE}/knowledge", timeout=10)
print("2. /knowledge:", r.status_code, r.text[:500])

# 3. 查询（纯文字）
payload = {"question": "服务器地址是什么", "mode": "mix"}
print(f"3. /query (text) request body: {payload}")
r = requests.post(f"{BASE}/query", json=payload, timeout=300)
print("3. /query (text):", r.status_code)
if r.status_code == 200:
    data = r.json()
    print("   answer preview:", json.dumps(data, ensure_ascii=False)[:1200])
else:
    print("   error body:", r.text[:1500])
    sys.exit(1)

# 4. 查询（文字 + 图片）
img_path = Path("test_image.png")
if img_path.exists():
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "question": "这张图里有什么？结合知识库回答",
        "mode": "mix",
        "images": [f"data:image/png;base64,{b64}"],
    }
    print(f"4. /query (text+image) request body: images length={len(b64)}")
    r = requests.post(f"{BASE}/query", json=payload, timeout=300)
    print("4. /query (text+image):", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print("   answer preview:", json.dumps(data, ensure_ascii=False)[:1200])
    else:
        print("   error body:", r.text[:1500])
else:
    print("4. /query (text+image): 跳过 - 未找到 test_image.png")
