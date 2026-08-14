"""
PaddleOCR-VL cloud document parser.

Wraps the PaddleOCR-VL HTTP job API (/api/v2/ocr/jobs) so it can be used as a
RAG-Anything parser. The service accepts a local file or a URL, runs layout /
OCR parsing server-side, and returns a JSONL result where each line holds one
page of layoutParsingResults containing markdown, embedded images and rendered
output images.

The parser is registered as a custom parser named "paddlecloud" via
register_parser(), then selected through RAGAnythingConfig(parser=...) or the
PARSER=paddlecloud environment variable.

Configuration (environment variables, see PaddleCloudConfig):

    PADDLE_CLOUD_URL      default https://paddleocr.aistudio-app.com/api/v2/ocr/jobs
    PADDLE_CLOUD_TOKEN    API bearer token (required)
    PADDLE_CLOUD_MODEL    default PaddleOCR-VL-1.6
    PADDLE_CLOUD_POLL     poll interval seconds, default 5
    PADDLE_CLOUD_TIMEOUT  per-request timeout seconds, default 120
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests

from raganything.parser import Parser, register_parser

logger = logging.getLogger(__name__)


class PaddleCloudError(RuntimeError):
    """Raised when the PaddleOCR-VL cloud API returns an error."""


class PaddleCloudConfig:
    """Runtime configuration for PaddleCloudParser."""

    DEFAULT_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    DEFAULT_MODEL = "PaddleOCR-VL-1.6"

    def __init__(
        self,
        token: Optional[str] = None,
        model: Optional[str] = None,
        job_url: Optional[str] = None,
        poll_interval: float = 5.0,
        timeout: float = 120.0,
        use_chart_recognition: bool = False,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
    ) -> None:
        self.token = token or os.getenv("PADDLE_CLOUD_TOKEN", "")
        self.model = model or os.getenv("PADDLE_CLOUD_MODEL", self.DEFAULT_MODEL)
        self.job_url = (
            job_url or os.getenv("PADDLE_CLOUD_URL", self.DEFAULT_URL)
        ).rstrip("/")
        self.poll_interval = float(
            os.getenv("PADDLE_CLOUD_POLL", poll_interval) or poll_interval
        )
        self.timeout = float(os.getenv("PADDLE_CLOUD_TIMEOUT", timeout) or timeout)
        self.use_chart_recognition = use_chart_recognition
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping

    def optional_payload(self) -> Dict[str, bool]:
        return {
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useChartRecognition": self.use_chart_recognition,
        }

    def validate(self) -> None:
        if not self.token:
            raise PaddleCloudError(
                "PaddleOCR-VL cloud token is not configured. "
                "Set PADDLE_CLOUD_TOKEN in the environment or pass token= to "
                "PaddleCloudConfig."
            )


class PaddleCloudClient:
    """Minimal HTTP client for the PaddleOCR-VL job API."""

    def __init__(self, config: PaddleCloudConfig) -> None:
        self.config = config
        self.config.validate()
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": "Bearer " + self.config.token}
        )

    def _headers(self, json_body: bool) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def submit(self, file_path: Union[str, Path]) -> str:
        path = str(file_path)
        if path.startswith("http://") or path.startswith("https://"):
            payload = {
                "fileUrl": path,
                "model": self.config.model,
                "optionalPayload": self.config.optional_payload(),
            }
            resp = self._session.post(
                self.config.job_url,
                json=payload,
                headers=self._headers(json_body=True),
                timeout=self.config.timeout,
            )
        else:
            if not os.path.exists(path):
                raise FileNotFoundError("File not found: " + path)
            data = {
                "model": self.config.model,
                "optionalPayload": json.dumps(self.config.optional_payload()),
            }
            with open(path, "rb") as fh:
                files = {"file": fh}
                resp = self._session.post(
                    self.config.job_url,
                    data=data,
                    files=files,
                    timeout=self.config.timeout,
                )
        return self._unwrap_job_id(resp)

    @staticmethod
    def _unwrap_job_id(resp: requests.Response) -> str:
        if resp.status_code != 200:
            raise PaddleCloudError(
                "Submit failed (%d): %s" % (resp.status_code, resp.text[:1000])
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise PaddleCloudError("Invalid JSON response: " + resp.text[:500]) from exc
        if body.get("code") not in (0, None):
            raise PaddleCloudError("Submit error: " + str(body))
        return body["data"]["jobId"]

    def poll(self, job_id: str, on_progress=None) -> str:
        url = self.config.job_url + "/" + job_id
        while True:
            resp = self._session.get(url, timeout=self.config.timeout)
            if resp.status_code != 200:
                raise PaddleCloudError(
                    "Poll failed (%d): %s" % (resp.status_code, resp.text[:1000])
                )
            data = resp.json().get("data", {})
            state = data.get("state")
            progress = data.get("extractProgress", {}) or {}
            total = progress.get("totalPages")
            extracted = progress.get("extractedPages")
            if on_progress:
                on_progress(state, total, extracted)
            if state == "done":
                result_url = (data.get("resultUrl") or {}).get("jsonUrl")
                if not result_url:
                    raise PaddleCloudError("Job done but no resultUrl returned.")
                return result_url
            if state == "failed":
                raise PaddleCloudError(
                    "Job failed: " + str(data.get("errorMsg", "unknown error"))
                )
            time.sleep(self.config.poll_interval)


def _is_url(value: str) -> bool:
    parsed = urlparse(str(value))
    return bool(parsed.scheme in ("http", "https") and parsed.netloc)


def _download(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _safe_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    return base or "asset"


def _heading_level(line: str):
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None, None
    count = 0
    for ch in stripped:
        if ch == "#":
            count += 1
        else:
            break
    if count == 0 or count > 6 or (len(stripped) > count and stripped[count] != " "):
        return None, None
    return count, stripped[count:].strip()


def _html_table_to_rows(html: str) -> List[List[str]]:
    """Parse an HTML <table> into a list of cell rows using html.parser.

    Handles <tr>/<th>/<td>, nested tables are flattened into the cell text,
    and the original cell order is preserved.
    """
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.rows: List[List[str]] = []
            self._current_row: Optional[List[str]] = None
            self._cell_parts: Optional[List[str]] = None
            self._depth = 0

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag == "tr":
                self._current_row = []
            elif tag in ("td", "th") and self._current_row is not None:
                self._cell_parts = []
                self._depth = 0
            elif self._cell_parts is not None:
                self._depth += 1

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in ("td", "th") and self._cell_parts is not None:
                cell = " ".join("".join(self._cell_parts).split())
                if self._current_row is not None:
                    self._current_row.append(cell)
                self._cell_parts = None
            elif tag == "tr" and self._current_row is not None:
                if self._current_row:
                    self.rows.append(self._current_row)
                self._current_row = None
            elif self._cell_parts is not None and self._depth > 0:
                self._depth -= 1

        def handle_data(self, data):
            if self._cell_parts is not None:
                self._cell_parts.append(data)

    parser = _TableParser()
    parser.feed(html)
    parser.close()
    return parser.rows


def _rows_to_markdown(rows: List[List[str]]) -> str:
    """Render rows as a GitHub-style Markdown table."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normalized = [r + [""] * (width - len(r)) for r in rows]

    def esc(cell: str) -> str:
        return str(cell).replace("|", "\\|").replace("\n", " ").strip()

    header = normalized[0]
    lines = ["| " + " | ".join(esc(c) for c in header) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in normalized[1:]:
        lines.append("| " + " | ".join(esc(c) for c in row) + " |")
    return "\n".join(lines)


def _build_section_path(stack):
    return " > ".join(title for _, title in stack if title)


def _looks_like_caption(text):
    import re

    if not text or len(text) > 120:
        return False
    return bool(
        re.match(
            r"^\s*(?:图\s*\d|Figure\s+\d|Fig\.?\s*\d|表\s*\d|Table\s+\d)",
            text,
            re.IGNORECASE,
        )
    )


def _looks_like_footnote(text):
    import re

    if not text or len(text) > 240:
        return False
    return bool(
        re.match(
            r"^\s*(?:注[：:]|备注[：:]|Notes?\s*[:：]|Source\s*[:：]|资料来源[：:]|数据来源[：:])",
            text,
            re.IGNORECASE,
        )
    )


def _tail_neighbor_text(blocks, max_chars=500):
    parts = []
    total = 0
    for block in reversed(blocks):
        if block.get("type") != "text":
            break
        text = (block.get("text") or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars and parts:
            break
        parts.insert(0, text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(parts).strip()


def _head_neighbor_text(lines, start_idx, max_chars=500):
    parts = []
    total = 0
    i = start_idx
    while i < len(lines):
        stripped = lines[i].rstrip().strip()
        if not stripped:
            if parts:
                break
            i += 1
            continue
        if (
            stripped.startswith("#")
            or stripped.startswith("|")
            or stripped.startswith("![")
            or "<table" in stripped.lower()
        ):
            break
        if _looks_like_caption(stripped) and parts:
            break
        parts.append(stripped)
        total += len(stripped)
        if total >= max_chars:
            break
        i += 1
    return "\n".join(parts).strip()


def markdown_to_content_list(
    markdown: str,
    images: Dict[str, str],
    assets_dir: Path,
    page_idx: int,
    text_threshold: int = 4,
) -> List[Dict[str, Any]]:
    """Convert one page of PaddleOCR-VL markdown into MinerU-style blocks.

    Image references (``![alt](images/xxx.jpg)``) are split out as image blocks
    and downloaded locally. A run of Markdown table lines or an HTML
    ``<table>`` (single-line or multi-line) becomes a single table block whose
    ``table_body`` is normalized to Markdown. Everything else is emitted as
    text blocks (headings carry ``text_level``).
    """
    content_list: List[Dict[str, Any]] = []
    lines = (markdown or "").splitlines()

    heading_stack: List[tuple] = []

    def flush_text(buf):
        if not buf:
            return
        joined = "\n".join(buf).strip()
        if joined:
            content_list.append({"type": "text", "text": joined, "page_idx": int(page_idx)})

    pending_text: List[str] = []
    table_lines: List[str] = []
    html_table_lines: List[str] = []

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        i += 1

        # Accumulate multi-line HTML tables (<table ...> ... </table>).
        if html_table_lines:
            html_table_lines.append(line)
            if "</table" in line.lower():
                html = "\n".join(html_table_lines)
                rows = _html_table_to_rows(html)
                body = _rows_to_markdown(rows)
                if not body:
                    body = html
                flush_text(pending_text)
                pending_text = []
                content_list.append(
                    {"type": "table", "table_body": body, "page_idx": int(page_idx)}
                )
                html_table_lines = []
            continue

        lower = line.lower()
        if "<table" in lower:
            if table_lines:
                body = "\n".join(table_lines)
                if len(body.splitlines()) >= text_threshold:
                    flush_text(pending_text)
                    pending_text = []
                    content_list.append(
                        {"type": "table", "table_body": body, "page_idx": int(page_idx)}
                    )
                else:
                    pending_text.extend(table_lines)
                table_lines = []
            if "</table" in lower:
                rows = _html_table_to_rows(line)
                body = _rows_to_markdown(rows) or line
                flush_text(pending_text)
                pending_text = []
                content_list.append(
                    {"type": "table", "table_body": body, "page_idx": int(page_idx)}
                )
            else:
                html_table_lines = [line]
            continue

        is_table = line.lstrip().startswith("|") and line.rstrip().endswith("|")

        if is_table:
            table_lines.append(line)
            continue
        if table_lines:
            body = "\n".join(table_lines)
            if len(body.splitlines()) >= text_threshold:
                flush_text(pending_text)
                pending_text = []
                content_list.append(
                    {"type": "table", "table_body": body, "page_idx": int(page_idx)}
                )
            else:
                pending_text.extend(table_lines)
            table_lines = []

        stripped = line.strip()
        if not stripped:
            pending_text.append(line)
            continue

        level, title = _heading_level(line)
        if level is not None and title and len(title) <= 80:
            flush_text(pending_text)
            pending_text = []
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            content_list.append(
                {
                    "type": "text",
                    "text": title,
                    "text_level": level,
                    "page_idx": int(page_idx),
                }
            )
            continue

        img_block = _parse_image_line(stripped, images, assets_dir, page_idx)
        if img_block is not None:
            flush_text(pending_text)
            pending_text = []

            above_text = _tail_neighbor_text(content_list, max_chars=500)

            captions: List[str] = []
            footnotes: List[str] = []
            j = i
            blank_seen = False
            while j < n:
                peek = lines[j].strip()
                if not peek:
                    blank_seen = True
                    j += 1
                    continue
                if not captions and _looks_like_caption(peek):
                    captions.append(peek)
                    j += 1
                    blank_seen = False
                    continue
                if captions and _looks_like_footnote(peek):
                    footnotes.append(peek)
                    j += 1
                    blank_seen = False
                    continue
                break
            i = j

            below_text = _head_neighbor_text(lines, i, max_chars=500)

            if captions:
                img_block["image_caption"] = list(captions)
                img_block["img_caption"] = list(captions)
            if footnotes:
                img_block["image_footnote"] = list(footnotes)
                img_block["img_footnote"] = list(footnotes)

            section_path = _build_section_path(heading_stack)
            if section_path:
                img_block["_section_path"] = section_path

            neighbor_parts: List[str] = []
            if above_text:
                neighbor_parts.append("[Above]\n" + above_text)
            if below_text:
                neighbor_parts.append("[Below]\n" + below_text)
            if neighbor_parts:
                img_block["_neighbor_text"] = "\n\n".join(neighbor_parts)

            content_list.append(img_block)
            continue

        pending_text.append(line)

    if table_lines:
        body = "\n".join(table_lines)
        if len(body.splitlines()) >= text_threshold:
            flush_text(pending_text)
            pending_text = []
            content_list.append(
                {"type": "table", "table_body": body, "page_idx": int(page_idx)}
            )
        else:
            pending_text.extend(table_lines)
    if html_table_lines:
        # Table never closed; keep it as text so nothing is silently dropped.
        pending_text.extend(html_table_lines)
    flush_text(pending_text)
    return content_list


_IMAGE_RE = None


def _parse_image_line(
    line: str,
    images: Dict[str, str],
    assets_dir: Path,
    page_idx: int,
) -> Optional[Dict[str, Any]]:
    global _IMAGE_RE
    if _IMAGE_RE is None:
        import re

        _IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

    stripped = line.strip()
    match = _IMAGE_RE.fullmatch(stripped)
    if not match:
        return None

    alt = (match.group(1) or "").strip()
    ref = match.group(2).strip()
    if not ref:
        return None

    candidates = [ref]
    if ref in images:
        candidates.append(images[ref])
    base = os.path.basename(ref)
    for key, url in images.items():
        if os.path.basename(key) == base or key.endswith(ref):
            candidates.append(url)

    url = next((c for c in candidates if _is_url(c)), None)
    if url is None:
        logger.warning("Could not resolve image URL for reference: %s", ref)
        return None

    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        local_name = "p%04d_%s" % (int(page_idx), _safe_name(ref))
        local_path = assets_dir / local_name
        _download(url, local_path)
    except Exception as exc:
        logger.warning("Failed to download image %s: %s", url, exc)
        return None

    return {
        "type": "image",
        "img_path": str(local_path),
        "image_caption": [alt] if alt else [],
        "img_caption": [alt] if alt else [],
        "page_idx": int(page_idx),
    }


class PaddleCloudParser(Parser):
    """RAG-Anything parser backed by the PaddleOCR-VL cloud job API."""

    PARSER_NAME = "paddlecloud"

    def __init__(self, config: Optional[PaddleCloudConfig] = None) -> None:
        super().__init__()
        self.config = config or PaddleCloudConfig()

    def check_installation(self) -> bool:
        try:
            self.config.validate()
        except PaddleCloudError:
            return False
        return True

    def _coerce_to_pdf(
        self, file_path: Path, output_dir: Path
    ) -> Optional[Path]:
        """Convert a format the cloud API does not accept directly to PDF.

        The PaddleOCR-VL cloud endpoint handles PDF, images and Office docs,
        but returns ``code 10004 (file format not supported)`` for plain text
        files (.txt, .md). The base Parser class ships a ReportLab-based
        converter, so we turn those files into PDF before submitting the job.
        Returns the generated PDF path, or None when the file can be sent as-is.
        """
        ext = file_path.suffix.lower()
        if ext not in self.TEXT_FORMATS:
            return None
        logger.info(
            "Converting unsupported format '%s' to PDF for cloud submission", ext
        )
        try:
            pdf_path = self.convert_text_to_pdf(str(file_path), str(output_dir))
            logger.info("Converted %s -> %s", file_path.name, pdf_path.name)
            return pdf_path
        except Exception as exc:
            raise RuntimeError(
                "Failed to convert " + str(file_path) + " to PDF for cloud parsing: "
                + str(exc)
            ) from exc

    def _progress(self, state, total, extracted):
        if state == "pending":
            logger.info("PaddleOCR-VL job is pending...")
        elif state == "running":
            if total is not None and extracted is not None:
                logger.info(
                    "PaddleOCR-VL parsing: %s/%s pages", extracted, total
                )
            else:
                logger.info("PaddleOCR-VL job running...")

    def _fetch_pages(self, jsonl_url: str) -> List[Dict[str, Any]]:
        resp = requests.get(jsonl_url, timeout=self.config.timeout)
        resp.raise_for_status()
        pages: List[Dict[str, Any]] = []
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line: %s", exc)
        return pages

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        del method, lang
        file_path = Path(file_path)
        if not str(file_path).startswith(("http://", "https://")) and not file_path.exists():
            raise FileNotFoundError("File does not exist: " + str(file_path))

        base_dir = (
            self._unique_output_dir(output_dir, file_path)
            if output_dir
            else file_path.parent / "paddlecloud_output"
        )
        assets_dir = Path(base_dir) / "images"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # The cloud API rejects formats like .txt/.md; convert them to PDF first.
        converted = self._coerce_to_pdf(file_path, Path(base_dir))
        if converted is not None:
            file_path = Path(converted)

        client = PaddleCloudClient(self.config)
        logger.info("Submitting PaddleOCR-VL job for: %s", file_path)
        job_id = client.submit(file_path)
        logger.info("PaddleOCR-VL job id: %s", job_id)
        jsonl_url = client.poll(job_id, on_progress=self._progress)
        logger.info("Downloading result: %s", jsonl_url)

        pages = self._fetch_pages(jsonl_url)
        content_list: List[Dict[str, Any]] = []
        global_page = 0
        for page in pages:
            results = (page.get("result") or {}).get("layoutParsingResults") or []
            for res in results:
                markdown = ((res.get("markdown") or {}).get("text")) or ""
                images = ((res.get("markdown") or {}).get("images")) or {}
                blocks = markdown_to_content_list(
                    markdown=markdown,
                    images=images,
                    assets_dir=assets_dir,
                    page_idx=global_page,
                )
                if not blocks and markdown.strip():
                    blocks = [
                        {
                            "type": "text",
                            "text": markdown.strip(),
                            "page_idx": global_page,
                        }
                    ]
                content_list.extend(blocks)
                global_page += 1

        # When the original input itself is an image (not a PDF), append it as
        # a dedicated image block so the multimodal pipeline can invoke the VLM
        # on it. The PaddleOCR-VL result for a raw image only yields OCR text;
        # without this block, multimodal_items would be empty and the image
        # signal would never reach the knowledge graph or VLM-enhanced query.
        is_image = (
            file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
        )
        if is_image:
            logger.info(
                "Appending raw image block for VLM processing: %s",
                file_path,
            )
            content_list.append(
                {
                    "type": "image",
                    "img_path": str(file_path),
                    "page_idx": 0,
                }
            )

        if not content_list:
            raise ValueError("PaddleOCR-VL returned no content for: " + str(file_path))
        logger.info("PaddleOCR-VL produced %d blocks", len(content_list))
        return content_list

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return self.parse_document(
            file_path=pdf_path, method=method, output_dir=output_dir, lang=lang, **kwargs
        )

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return self.parse_document(
            file_path=image_path, method="auto", output_dir=output_dir, lang=lang, **kwargs
        )

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        # The cloud API can ingest Office files directly; pass it through.
        return self.parse_document(
            file_path=doc_path, method="auto", output_dir=output_dir, lang=lang, **kwargs
        )


def register() -> None:
    """Register the PaddleCloudParser under the name ``paddlecloud``."""
    register_parser(PaddleCloudParser.PARSER_NAME, PaddleCloudParser)


# Auto-register on import so selecting parser="paddlecloud" just works.
register()
