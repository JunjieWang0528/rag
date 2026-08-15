import requests
import time
import sys

BASE = "http://localhost:8000"
FILES = [
    r"E:\Desktop\xue_xi\rag-anything\uploads\1786162041_a84597_test_upload.txt",
    r"E:\Desktop\xue_xi\rag-anything\uploads\1786264967_88df13_服务器地址.docx",
    r"E:\Desktop\xue_xi\rag-anything\uploads\1786266183_90366f_服务器地址.txt",
]

for fp in FILES:
    print(f"\n=== UPLOADING: {fp.split(chr(92))[-1]} ===")
    with open(fp, "rb") as f:
        r = requests.post(f"{BASE}/upload", files={"files": f}, timeout=30)
    print(f"  upload resp: {r.status_code}")
    data = r.json()
    task_id = data.get("task_id")
    if not task_id:
        print(f"  ERROR: {data}")
        sys.exit(1)
    print(f"  task_id={task_id}")
    # poll until done
    for i in range(120):
        t = requests.get(f"{BASE}/task/{task_id}", timeout=10).json()
        status = t.get("status", "?")
        if status in ("done", "completed", "processed"):
            print(f"  DONE after {i+1}s: {t.get('message','')}")
            break
        time.sleep(3)
    else:
        print(f"  TIMEOUT, last status: {t}")

print("\n=== DONE ===")
