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
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# ---------- 从 .env 读取火山方舟配置 ----------
API_KEY = os.getenv("LLM_BINDING_API_KEY")
BASE_URL = os.getenv("LLM_BINDING_HOST")
LLM_MODEL = os.getenv("LLM_MODEL")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")


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
    """视觉模型调用函数（用于图片理解，和 LLM 用同一个模型）"""
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
        embedding_dim=2048,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=EMB_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
        ),
    )


async def main(doc_paths):
    if not doc_paths:
        print("❌ 请提供文档路径，例如：python rag_demo.py 我的文档.pdf")
        return

    # 1. 初始化 RAGAnything（working_dir 是知识库存储位置）
    config = RAGAnythingConfig(
        working_dir="./rag_storage",   # 知识库数据存这里
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,   # 处理图片
        enable_table_processing=True,   # 处理表格
        enable_equation_processing=True, # 处理公式
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func(),
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