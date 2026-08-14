import requests, json, sys

BASE = "http://localhost:8000"

# 1. 健康检查
r = requests.get(f"{BASE}/health", timeout=10)
print("1. /health:", r.status_code, r.text[:200])

# 2. 文档列表
r = requests.get(f"{BASE}/knowledge", timeout=10)
print("2. /knowledge:", r.status_code, r.text[:500])

# 3. 查询（仅传 question + mode，匹配真实 QueryRequest schema）
payload = {"question": "服务器地址是什么", "mode": "mix"}
print(f"3. /query request body: {payload}")
r = requests.post(f"{BASE}/query", json=payload, timeout=300)
print("3. /query:", r.status_code)
if r.status_code == 200:
    data = r.json()
    print("   answer preview:", json.dumps(data, ensure_ascii=False)[:1200])
else:
    print("   error body:", r.text[:1500])
    sys.exit(1)
