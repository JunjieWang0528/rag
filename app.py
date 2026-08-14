"""
RAG-Anything Web 应用
=====================
通过浏览器上传文档/图片 -> 自动存入知识库 -> RAG 检索回答

启动方式：
    conda activate rag-anything
    uvicorn app:app --host 0.0.0.0 --port 9621

或直接（端口由 .env 的 HOST / PORT 决定，默认 9621）：
    python app.py
"""
import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# 必须在导入 raganything 之前加载 .env（保证 tiktoken 缓存目录等配置生效）
load_dotenv()

from raganything import RAGAnything, RAGAnythingConfig
from raganything.parser_paddle_cloud import PaddleCloudParser  # noqa: F401 (registers "paddlecloud")
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
WORKING_DIR = BASE_DIR / "rag_storage"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WORKING_DIR.mkdir(exist_ok=True)

# 支持的扩展名（与 config 中默认值保持一致）
SUPPORTED_EXT = {
    ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".gif", ".webp", ".doc", ".docx", ".ppt", ".pptx",
    ".xls", ".xlsx", ".txt", ".md",
}

# ---------------------------------------------------------------------------
# LLM / Embedding 配置（参考 rag_demo.py）
# ---------------------------------------------------------------------------
API_KEY = os.getenv("LLM_BINDING_API_KEY")
BASE_URL = os.getenv("LLM_BINDING_HOST")
LLM_MODEL = os.getenv("LLM_MODEL")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMB_DIM = int(os.getenv("EMBEDDING_DIM", "2048"))

# Document parser: "paddlecloud" (PaddleOCR-VL cloud API) by default,
# overridable via the PARSER environment variable.
DOC_PARSER = os.getenv("PARSER", "paddlecloud")


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM 调用函数（豆包 doubao-seed-evolving）"""
    return openai_complete_if_cache(
        LLM_MODEL, prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
    """视觉模型调用函数（用于图片理解）"""
    if messages:
        return openai_complete_if_cache(
            LLM_MODEL, "", system_prompt=None, history_messages=[],
            messages=messages, api_key=API_KEY, base_url=BASE_URL, **kwargs,
        )
    elif image_data:
        return openai_complete_if_cache(
            LLM_MODEL, "", system_prompt=None, history_messages=[],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            }],
            api_key=API_KEY, base_url=BASE_URL, **kwargs,
        )
    else:
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)


def embedding_func():
    """Embedding 函数（doubao-embedding-vision）"""
    return EmbeddingFunc(
        embedding_dim=EMB_DIM,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=EMB_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
        ),
    )


# ---------------------------------------------------------------------------
# 全局单例 RAGAnything
# ---------------------------------------------------------------------------
_rag = None
_rag_lock = threading.Lock()


def get_rag() -> RAGAnything:
    """获取全局 RAGAnything 单例（懒初始化）"""
    global _rag
    with _rag_lock:
        if _rag is None:
            config = RAGAnythingConfig(
                working_dir=str(WORKING_DIR),
                parser=DOC_PARSER,
                parse_method="auto",
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )
            _rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func(),
            )
        return _rag


# ---------------------------------------------------------------------------
# 任务状态跟踪（简单的进程内存储）
# ---------------------------------------------------------------------------
class TaskStore:
    """记录每个上传任务的处理状态"""

    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()

    def create(self):
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "status": "pending",          # pending / processing / done / failed
                "message": "排队中",
                "files": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "error": None,
            }
        return task_id

    def get(self, task_id):
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id, **kwargs):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.update(kwargs)
            task["updated_at"] = datetime.now().isoformat()

    def all(self):
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t["created_at"], reverse=True)


tasks = TaskStore()


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG-Anything Web", description="多模态 RAG 文档问答")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_load_index_html())


def _load_index_html() -> str:
    """读取前端 HTML 文件内容"""
    html_path = BASE_DIR / "web" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return f"<h1>未找到前端页面 web/index.html</h1>"


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


async def _process_files_async(task_id: str, save_paths):
    """在后台任务中处理上传的文件"""
    try:
        tasks.update(task_id, status="processing", message="开始处理文档...")
        rag = get_rag()

        done_files = []
        for save_path in save_paths:
            file_name = Path(save_path).name
            tasks.update(
                task_id,
                status="processing",
                message=f"正在解析并存入知识库: {file_name}",
            )
            await rag.process_document_complete(
                file_path=save_path,
                output_dir=str(OUTPUT_DIR),
                parse_method="auto",
            )
            done_files.append(file_name)

        tasks.update(
            task_id,
            status="done",
            message="所有文档处理完成，可以提问了！",
            files=done_files,
        )
    except Exception as exc:
        tasks.update(
            task_id,
            status="failed",
            message="处理失败",
            error=str(exc),
        )


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """接收多个文件，保存到本地并异步处理"""
    if not files:
        raise HTTPException(status_code=400, detail="未收到任何文件")

    # 校验扩展名
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {f.filename}（支持: {', '.join(sorted(SUPPORTED_EXT))}）",
            )

    task_id = tasks.create()

    # 保存文件
    save_paths = []
    for f in files:
        safe_name = Path(f.filename).name
        # 防止重名覆盖
        dest = UPLOAD_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{safe_name}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        save_paths.append(str(dest))

    # 后台异步处理，避免阻塞请求
    asyncio.get_running_loop().create_task(_process_files_async(task_id, save_paths))

    return JSONResponse({
        "task_id": task_id,
        "status": "accepted",
        "message": f"已接收 {len(files)} 个文件，处理中...",
        "files": [Path(p).name for p in save_paths],
    })


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    """查询任务处理状态"""
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.get("/files")
async def list_files():
    """列出已上传的文件"""
    results = []
    for p in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file():
            results.append({
                "name": p.name,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
    return {"files": results}


@app.get("/knowledge")
async def list_knowledge():
    """列出知识库中已入库（已处理）的文档，重启后依然存在"""
    doc_status_file = WORKING_DIR / "kv_store_doc_status.json"
    docs = []
    if doc_status_file.exists():
        try:
            data = json.loads(doc_status_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        for doc_id, meta in data.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("status") not in ("processed", "done", "completed"):
                continue
            docs.append({
                "doc_id": doc_id,
                "name": meta.get("file_path", doc_id),
                "status": meta.get("status", "processed"),
                "chunks_count": meta.get("chunks_count", 0),
                "content_length": meta.get("content_length", 0),
                "created_at": meta.get("created_at", ""),
                "content_summary": (meta.get("content_summary", "") or "")[:200],
            })
    return {"docs": docs, "count": len(docs)}


@app.delete("/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str):
    """从知识库中删除指定文档及其关联的 chunks / entities / relationships"""
    doc_status_file = WORKING_DIR / "kv_store_doc_status.json"
    docs = {}
    if doc_status_file.exists():
        try:
            docs = json.loads(doc_status_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    if doc_id not in docs:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

    try:
        rag = get_rag()
        # LightRAG is lazily initialized on first document upload; ensure it's
        # ready before we try to delete from it.
        init_result = await rag._ensure_lightrag_initialized()
        if init_result and not init_result.get("success"):
            raise RuntimeError(f"LightRAG init failed: {init_result.get('error')}")
        result = await rag.lightrag.adelete_by_doc_id(doc_id)
        name = docs[doc_id].get("file_path", doc_id)
        return {"ok": True, "doc_id": doc_id, "name": name, "detail": str(result)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除失败: {exc}")


@app.post("/query")
async def query(req: QueryRequest):
    """RAG 检索回答"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        rag = get_rag()
        result = await rag.aquery(question, mode=req.mode)
        return {"answer": result, "mode": req.mode, "question": question}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询失败: {exc}")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9621"))
    uvicorn.run(app, host=host, port=port)