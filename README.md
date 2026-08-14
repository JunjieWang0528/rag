# RAG Enhanced Fork

基于 [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) 的增强版本。

本项目在 [RAG-Anything](https://github.com/HKUDS/RAG-Anything)（基于 [LightRAG](https://github.com/HKUDS/LightRAG) 的多模态 RAG 框架）之上进行扩展，保留原有能力，并新增以下功能。

---

## 扩展功能

### 1. PaddleOCR-VL 云端文档解析器 (`parser=paddlecloud`)

新增基于 [PaddleOCR-VL](https://paddleocr.aistudio-app.com) 云端 API 的文档解析器，无需本地 GPU。

- 支持 PDF、图片、Office 文档（.docx / .pptx / .xlsx）
- 自动将 `.txt` / `.md` 等不支持格式转为 PDF 后提交
- **增强图片元数据提取**：自动识别图注（`图 3-1 …` / `Figure 2.1`）、脚注（`注：` / `Source:`）、章节路径（`_section_path`）、上下邻近正文（`_neighbor_text`），提升图片召回质量
- 产出与 MinerU 兼容的 `content_list`，可无缝接入原有多模态流水线

```bash
# 环境变量
PADDLE_CLOUD_TOKEN=your-api-token
PADDLE_CLOUD_MODEL=PaddleOCR-VL-1.6   # 可选
PARSER=paddlecloud
```

```python
from raganything import RAGAnything, RAGAnythingConfig

config = RAGAnythingConfig(
    working_dir="./rag_storage",
    parser="paddlecloud",
)
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `PADDLE_CLOUD_TOKEN` | （必填） | API Token |
| `PADDLE_CLOUD_MODEL` | `PaddleOCR-VL-1.6` | 模型版本 |
| `PADDLE_CLOUD_URL` | `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs` | API 地址 |
| `PADDLE_CLOUD_POLL` | `5` | 轮询间隔（秒） |
| `PADDLE_CLOUD_TIMEOUT` | `120` | 请求超时（秒） |

### 2. VLM 增强查询（召回图片后回传视觉模型）

查询时先用 LightRAG 召回相关 chunk；若 chunk 中含 `Image Path:`，自动从磁盘读取图片、编码为 base64，按原位置插入多模态消息，再交给视觉大模型综合文本与图片作答。

```python
# 配置 vision_model_func 后，查询自动启用 VLM 增强
result = await rag.aquery(
    "图中服务器的 IP 地址是多少？",
    mode="hybrid",
    vlm_enhanced=True,
)
```

### 3. Web API 服务 (`app.py`)

基于 FastAPI 的 Web 服务，支持文档上传、解析入库与 VLM 多模态问答。

```bash
python app.py
# 默认监听 http://0.0.0.0:8000
```

### 4. 命令行演示工具 (`rag_demo.py`)

交互式命令行 RAG 演示，方便本地快速验证。

```bash
python rag_demo.py
```

### 5. 辅助脚本

| 脚本 | 用途 |
|---|---|
| `e2e_test.py` | 端到端流水线测试 |
| `probe_query.py` | 查询探针，调试召回结果 |
| `reupload.py` | 文档重新解析 / 入库 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/JunjieWang0528/rag.git
cd rag

# 推荐使用 uv
uv sync
# 或
pip install -e .
```

### 配置

复制并编辑 `.env`（**不要提交含真实 Key 的 `.env`**）：

```bash
# LLM
LLM_BINDING=openai
LLM_MODEL=your-model
LLM_BINDING_HOST=https://your-api-host/v1
LLM_BINDING_API_KEY=your-key

# Embedding
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIM=2048
EMBEDDING_BINDING_HOST=https://your-api-host/v1
EMBEDDING_BINDING_API_KEY=your-key

# 解析器
PARSER=paddlecloud
PADDLE_CLOUD_TOKEN=your-paddle-token
```

### 处理文档并查询

```python
import asyncio
from raganything import RAGAnything, RAGAnythingConfig
# ... 配置 llm_model_func / vision_model_func / embedding_func ...

async def main():
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="paddlecloud",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    await rag.process_document_complete(
        file_path="path/to/document.pdf",
        output_dir="./output",
    )

    result = await rag.aquery("你的问题", mode="hybrid")
    print(result)

asyncio.run(main())
```

更完整的用法、多模态处理器、知识图谱构建等，请参阅上游文档：  
→ [HKUDS/RAG-Anything README](https://github.com/HKUDS/RAG-Anything)

---

## 与上游的关系

| | 上游 RAG-Anything | 本仓库 |
|---|---|---|
| 解析器 | MinerU / Docling / PaddleOCR（本地） | **+ PaddleOCR-VL 云端** |
| 图片元数据 | 依赖 parser 自带 caption | **自动提取图注 / 脚注 / 章节 / 邻近正文** |
| 查询 | VLM Enhanced Query | 保留并完善 marker → base64 流程 |
| 服务入口 | 无内置 Web 服务 | **`app.py` FastAPI** |
| 演示工具 | `examples/` | **+ `rag_demo.py` 等** |

本仓库定期可与上游同步；扩展代码主要集中在：

- `raganything/parser_paddle_cloud.py`
- `raganything/query.py`（VLM 图片回传）
- `app.py` / `rag_demo.py` / `e2e_test.py` / `probe_query.py` / `reupload.py`

---

## 致谢

- [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) — 多模态 RAG 框架
- [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) — 图谱增强检索
- [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) — 云端文档解析

```bibtex
@misc{guo2025raganythingallinoneragframework,
      title={RAG-Anything: All-in-One RAG Framework},
      author={Zirui Guo and Xubin Ren and Lingrui Xu and Jiahao Zhang and Chao Huang},
      year={2025},
      eprint={2510.12323},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.12323},
}
```

---

## License

遵循上游 [RAG-Anything](https://github.com/HKUDS/RAG-Anything) 的开源协议。
