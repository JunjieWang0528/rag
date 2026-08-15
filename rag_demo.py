"""
RAG-Anything 使用演示
流程：上传文档 -> 解析并存入知识库 -> RAG 检索

用法：
    python rag_demo.py <文档路径> [文档路径2 ...]
    支持 PDF / 图片 / txt 等格式

示例：
    python rag_demo.py "C:/docs/我的论文.pdf"
    python rag_demo.py "C:/docs/图片1.png" "C:/docs/报告.docx"
"""

import asyncio
import os
import sys
from functools import partial
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from raganything import RAGAnything, RAGAnythingConfig
from raganything.utils import query_instruct_ctx
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.rerank import ali_rerank
from lightrag.utils import EmbeddingFunc
import numpy as np
import dashscope
from http import HTTPStatus


# ---------- 从 .env 读取火山方舟配置 ----------
API_KEY = os.getenv("LLM_BINDING_API_KEY")
BASE_URL = os.getenv("LLM_BINDING_HOST")
LLM_MODEL = os.getenv("LLM_MODEL")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_BINDING_API_KEY")


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM 调用函数（豆包 doubao-seed-evolving）"""
    return openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


def vision_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
):
    """视觉模型调用函数（用于图片理解，和 LLM 用同一个模型）"""
    if messages:
        return openai_complete_if_cache(
            LLM_MODEL,
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=API_KEY,
            base_url=BASE_URL,
            **kwargs,
        )
    elif image_data:
        return openai_complete_if_cache(
            LLM_MODEL,
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
            api_key=API_KEY,
            base_url=BASE_URL,
            **kwargs,
        )
    else:
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)


async def _qwen3_vl_embed(
    texts: list[str], api_key: str | None = None, **kwargs
) -> np.ndarray:
    """使用 DashScope SDK 调用 qwen3-vl-embedding，enable_fusion=true，1024维

    自动检测文本中的 Image Path: 行，若有则读取图片文件并作为 image 参数传入，
    实现真正的图文融合向量嵌入。

    注意：enable_fusion=true 时 dashscope 会把整个 input 列表融合成一个向量，
    因此必须逐条调用 API，不能批量。
    """
    import base64
    import re

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    IMG_RE = re.compile(
        r"Image Path: (.+\.(png|jpg|jpeg|gif|webp|bmp|tiff|svg))", re.IGNORECASE
    )
    MIME_MAP = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "svg": "image/svg+xml",
    }

    async def _call_one(text: str) -> list[float]:
        m = IMG_RE.search(text)
        if m:
            img_path = m.group(1)
            clean_text = IMG_RE.sub("", text)
            try:
                with open(img_path, "rb") as f:
                    raw = f.read()
                ext = img_path.rsplit(".", 1)[-1].lower()
                img_b64 = "data:{};base64,{}".format(
                    MIME_MAP.get(ext, "image/png"),
                    base64.b64encode(raw).decode("utf-8"),
                )
                inp = [{"text": clean_text, "image": img_b64}]
            except FileNotFoundError:
                inp = [{"text": text}]
        else:
            inp = [{"text": text}]

        # 根据调用上下文自动选择 text_type：
        # LightRAG 在查询时传入 context="query"，索引时传入 context="document"
        text_type = "query" if kwargs.get("context") == "query" else "document"

        # 读取 LLM 生成的 instruct（查询时由 aquery 设置上下文变量）
        instruct = query_instruct_ctx.get() if text_type == "query" else None

        call_kwargs = dict(
            api_key=api_key,
            model="qwen3-vl-embedding",
            input=inp,
            text_type=text_type,
            enable_fusion=True,
            dimension=1024,
        )
        if instruct:
            call_kwargs["instruct"] = instruct

        resp = dashscope.MultiModalEmbedding.call(**call_kwargs)
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"Embedding API error: {resp}")
        return resp.output["embeddings"][0]["embedding"]

    results = await asyncio.gather(*[_call_one(t) for t in texts])
    return np.array(results, dtype=np.float32)


def embedding_func():
    """Embedding 函数（qwen3-vl-embedding，enable_fusion=true，1024维）"""
    return EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=partial(_qwen3_vl_embed, api_key=EMBEDDING_API_KEY),
    )


async def main(doc_paths):
    if not doc_paths:
        print("❌ 请提供文档路径，例如：python rag_demo.py 我的文档.pdf")
        return

    # 1. 初始化 RAGAnything（working_dir 是知识库存储位置）
    config = RAGAnythingConfig(
        working_dir="./rag_storage",  # 知识库数据存这里
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,  # 处理图片
        enable_table_processing=True,  # 处理表格
        enable_equation_processing=True,  # 处理公式
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func(),
        lightrag_kwargs={
            "kv_storage": os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
            "vector_storage": os.getenv(
                "LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"
            ),
            "doc_status_storage": os.getenv(
                "LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
            ),
            "graph_storage": os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
            "rerank_model_func": partial(
                ali_rerank,
                model="qwen3-vl-rerank",
                api_key=EMBEDDING_API_KEY,
            ),
        },
    )
    print("✅ 知识库初始化完成，存储目录：./rag_storage")

    # 2. 逐个上传并处理文档（解析 + 存入知识图谱）
    for p in doc_paths:
        path = Path(p)
        if not path.exists():
            print(f"❌ 文件不存在: {p}")
            continue
        print(f"\n📤 上传文档: {path.name}")
        await rag.process_document_complete(
            file_path=str(path.absolute()),
            output_dir="./output",
            parse_method="auto",
        )
        print(f"✅ 已处理并存入知识库: {path.name}")

    # 3. 进入交互式 RAG 检索
    print("\n" + "=" * 50)
    print("🔍 知识库已就绪，输入问题开始检索（输入 q 退出）")
    print("=" * 50)
    while True:
        question = input("\n❓ 提问: ").strip()
        if question.lower() in ("q", "quit", "exit", "退出"):
            break
        if not question:
            continue
        print("⏳ 检索中...")
        result = await rag.aquery(question, mode="hybrid")
        print(f"💬 回答: {result}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
