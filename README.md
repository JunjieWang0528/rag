# RAG Enhanced Fork

基于 [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) 的增强版本。

本项目在 [RAG-Anything](https://github.com/HKUDS/RAG-Anything)（基于 [LightRAG](https://github.com/HKUDS/LightRAG) 的多模态 RAG 框架）之上进行扩展，保留原有能力，并新增以下功能。

---

## 扩展功能

### 1. 阿里云百炼模型全栈支持

本项目深度集成阿里云百炼（DashScope）平台，支持以下模型：

| 功能 | 模型 | 说明 |
|------|------|------|
| **LLM / 视觉模型** | `qwen3.7-flash` | 多模态大模型，支持文本+图片理解 |
| **多模态 Embedding** | `qwen3-vl-embedding` | 图文融合向量，`enable_fusion=true`，1024 维 |
| **重排序 Rerank** | `qwen3-vl-rerank` | 多模态重排序，提升检索精度 |

#### 1a. 多模态融合 Embedding

文档入库时，PaddleOCR-VL 解析出的图片路径（`Image Path: xxx.png`）会被自动检测，图片与文本一起送入 `qwen3-vl-embedding` 生成融合向量，实现真正的图文联合检索。

- **`text_type` 自动区分**：查询时使用 `text_type="query"`，文档索引时使用 `text_type="document"`，提升检索匹配精度
- **`instruct` 动态生成**：查询时自动用 LLM（qwen3.7-flash）从用户问题中提取任务指令传给 embedding 模型，进一步优化向量质量（1-5% 精度提升）

#### 1b. 重排序 (Rerank)

检索阶段使用 `qwen3-vl-rerank` 对 LightRAG 召回的候选结果进行重排序，将最相关的内容排在前面，显著提升回答质量。

### 2. PaddleOCR-VL 云端文档解析器 (`parser=paddlecloud`)

新增基于 [PaddleOCR-VL](https://paddleocr.aistudio-app.com) 云端 API 的文档解析器，无需本地 GPU。

- 支持 PDF、图片、Office 文档（.docx / .pptx / .xlsx）
- 自动将 `.txt` / `.md` 等不支持格式转为 PDF 后提交
- **增强图片元数据提取**：自动识别图注（`图 3-1 …` / `Figure 2.1`）、脚注（`注：` / `Source:`）、章节路径（`_section_path`）、上下邻近正文（`_neighbor_text`），提升图片召回质量
- 产出与 MinerU 兼容的 `content_list`，可无缝接入原有多模态流水线

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `PADDLE_CLOUD_TOKEN` | （必填） | API Token |
| `PADDLE_CLOUD_MODEL` | `PaddleOCR-VL-1.6` | 模型版本 |
| `PADDLE_CLOUD_URL` | `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs` | API 地址 |
| `PADDLE_CLOUD_POLL` | `5` | 轮询间隔（秒） |
| `PADDLE_CLOUD_TIMEOUT` | `120` | 请求超时（秒） |

### 2. VLM 增强查询（召回图片后回传视觉模型）

查询时先用 LightRAG 召回相关 chunk；若 chunk 中含 `Image Path:`，自动从磁盘读取图片、编码为 base64，按原位置插入多模态消息，再交给视觉大模型综合文本与图片作答。

### 3. 用户查询支持传入图片

通过 `POST /query` API 的 `images` 字段（base64 编码），用户可以上传图片与文字组合提问。系统会同时检索知识库相关内容，并将用户图片一并送入视觉大模型，实现图文结合的综合问答。

### 4. Web API 服务 (`app.py`)

基于 FastAPI 的 Web 服务，支持文档上传、解析入库与 VLM 多模态问答。

### 5. 命令行演示工具 (`rag_demo.py`)

交互式命令行 RAG 演示，方便本地快速验证。

### 6. 辅助脚本

| 脚本 | 用途 |
|---|---|
| `e2e_test.py` | 端到端流水线测试 |
| `probe_query.py` | 查询探针，调试召回结果 |
| `reupload.py` | 文档重新解析 / 入库 |

---

## 快速开始

### 安装

推荐使用 `uv sync` 或 `pip install -e .` 安装依赖。

### 配置

复制并编辑 `.env`（**不要提交含真实 Key 的 `.env`**），配置 LLM、Embedding、解析器等环境变量。

### 处理文档并查询

配置 `llm_model_func` / `vision_model_func` / `embedding_func` 后，初始化 `RAGAnything`，调用 `process_document_complete()` 入库，然后通过 `aquery()` 检索问答。

更完整的用法、多模态处理器、知识图谱构建等，请参阅上游文档：
→ [HKUDS/RAG-Anything README](https://github.com/HKUDS/RAG-Anything)

---

## 与上游的关系

| | 上游 RAG-Anything | 本仓库 |
|---|---|---|
| 解析器 | MinerU / Docling / PaddleOCR（本地） | **+ PaddleOCR-VL 云端** |
| 图片元数据 | 依赖 parser 自带 caption | **自动提取图注 / 脚注 / 章节 / 邻近正文** |
| 查询 | VLM Enhanced Query | 保留并完善 marker → base64 流程 |
| 模型平台 | 通用 OpenAI 兼容接口 | **+ 阿里云百炼全栈（LLM / Embedding / Rerank）** |
| Embedding | 普通文本向量 | **多模态融合向量（text_type + instruct 优化）** |
| 重排序 | 无内置 | **qwen3-vl-rerank** |
| 用户图片查询 | 不支持 | **POST /query 支持 base64 图片上传** |
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

（引用上游论文：RAG-Anything: All-in-One RAG Framework, arXiv:2510.12323）

---

## License

遵循上游 [RAG-Anything](https://github.com/HKUDS/RAG-Anything) 的开源协议。
