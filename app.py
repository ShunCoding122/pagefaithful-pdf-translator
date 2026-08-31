"""Local-first, layout-aware PDF ebook translator."""

import asyncio
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import median
from threading import Lock

import pymupdf as fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from openai import APITimeoutError, OpenAI

APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "300"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TRANSLATION_WORKERS = int(os.getenv("TRANSLATION_WORKERS", "4"))
BATCH_CHAR_LIMIT = int(os.getenv("BATCH_CHAR_LIMIT", "12000"))

app = FastAPI(title="PageFaithful PDF Translator")
_progress_lock = Lock()
_progress = {"stage": "idle", "completed": 0, "total": 0, "pages": 0, "message": ""}


@dataclass
class TextBlock:
    rect: fitz.Rect
    text: str


def set_progress(stage: str, message: str, *, completed: int | None = None,
                 total: int | None = None, received: int | None = None, pages: int | None = None) -> None:
    with _progress_lock:
        _progress["stage"] = stage
        _progress["message"] = message
        if completed is not None:
            _progress["completed"] = completed
        if total is not None:
            _progress["total"] = total
        if received is not None:
            _progress["received"] = received
        if pages is not None:
            _progress["pages"] = pages


def advance_progress(amount: int) -> None:
    with _progress_lock:
        _progress["completed"] += amount


def add_received_characters(amount: int) -> None:
    with _progress_lock:
        _progress["received"] += amount


def openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            503,
            "OPENAI_API_KEY is not configured. Create .env from .env.example, then restart the app.",
        )
    # One retry handles a transient network fault without silently waiting through many retries.
    return OpenAI(max_retries=0, timeout=300.0)


def paragraph_blocks(page: fitz.Page) -> list[TextBlock]:
    """Turn a simple ebook page into one readable, paragraph-aware text region."""
    lines: list[TextBlock] = []
    for item in page.get_text("dict")["blocks"]:
        if item["type"] != 0:
            continue
        text = " ".join(
            "".join(span["text"] for span in line["spans"]).strip()
            for line in item["lines"]
        ).strip()
        if len(text) >= 2:
            lines.append(TextBlock(fitz.Rect(item["bbox"]), text))

    if not lines:
        return []
    lines.sort(key=lambda block: (round(block.rect.y0, 1), block.rect.x0))
    typical_height = median(max(1.0, line.rect.height) for line in lines)
    max_same_paragraph_gap = max(3.0, typical_height * 1.55)
    left_edge = min(line.rect.x0 for line in lines)

    pieces = [lines[0].text]
    for index, line in enumerate(lines[1:], start=1):
        previous = lines[index - 1]
        vertical_gap = line.rect.y0 - previous.rect.y1
        starts_paragraph = (
            vertical_gap > max_same_paragraph_gap
            or line.rect.x0 - left_edge > 8
        )
        pieces.append("\n\n" if starts_paragraph else " ")
        pieces.append(line.text)

    rect = fitz.Rect(lines[0].rect)
    for line in lines[1:]:
        rect |= line.rect
    return [TextBlock(rect, "".join(pieces))]


def streamed_output(client: OpenAI, **request) -> str:
    """Collect a long Responses API result as it arrives, avoiding one long idle read."""
    stream = client.responses.create(stream=True, store=False, **request)
    pieces: list[str] = []
    for event in stream:
        if event.type == "response.output_text.delta":
            pieces.append(event.delta)
            add_received_characters(len(event.delta))
        elif event.type == "error":
            raise RuntimeError(getattr(event, "message", "The translation service returned an error."))
    return "".join(pieces).strip()


def translate_one(client: OpenAI, text: str, language: str) -> str:
    prompt = (
        f"Translate the following private ebook paragraph into {language}. Preserve names, "
        "numbers, quotations, citations, and paragraph meaning. Return only the translation.\n\n"
        + text
    )
    return streamed_output(client, model=MODEL, input=prompt)


def translate_batch(client: OpenAI, texts: list[str], language: str) -> list[str]:
    schema = {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }
    payload = [{"id": index, "text": text} for index, text in enumerate(texts)]
    prompt = (
        f"Translate every paragraph in this JSON array into {language}. This is a private ebook. "
        "Keep every supplied id exactly once. Do not summarize or add commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    try:
        raw = streamed_output(
            client,
            model=MODEL,
            input=prompt,
            text={"format": {"type": "json_schema", "name": "paragraph_translations", "strict": True, "schema": schema}},
        )
    except APITimeoutError:
        # A congested or unusually long batch is split instead of terminating the book.
        if len(texts) > 1:
            midpoint = len(texts) // 2
            return (
                translate_batch(client, texts[:midpoint], language)
                + translate_batch(client, texts[midpoint:], language)
            )
        return [translate_one(client, texts[0], language)]

    try:
        items = json.loads(raw)["translations"]
    except (json.JSONDecodeError, KeyError, TypeError):
        items = []

    translated_by_id = {
        item["id"]: item["text"].strip()
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("id"), int)
        and 0 <= item["id"] < len(texts)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    }
    return [
        translated_by_id.get(index) or translate_one(client, text, language)
        for index, text in enumerate(texts)
    ]


def translate_document(
    client: OpenAI, page_blocks: list[list[TextBlock]], language: str
) -> dict[tuple[int, int], str]:
    jobs: list[tuple[list[tuple[int, int]], list[str]]] = []
    refs: list[tuple[int, int]] = []
    texts: list[str] = []
    chars = 0

    for page_no, blocks in enumerate(page_blocks):
        for block_no, block in enumerate(blocks):
            if texts and chars + len(block.text) > BATCH_CHAR_LIMIT:
                jobs.append((refs, texts))
                refs, texts, chars = [], [], 0
            refs.append((page_no, block_no))
            texts.append(block.text)
            chars += len(block.text)
    if texts:
        jobs.append((refs, texts))

    total = sum(len(blocks) for blocks in page_blocks)
    set_progress("translating", "正在翻译文字段", completed=0, total=total, received=0)

    translations: dict[tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=max(1, TRANSLATION_WORKERS)) as pool:
        pending = {
            pool.submit(translate_batch, client, batch_texts, language): batch_refs
            for batch_refs, batch_texts in jobs
        }
        for future in as_completed(pending):
            batch_refs = pending[future]
            batch_translations = future.result()
            translations.update(dict(zip(batch_refs, batch_translations)))
            advance_progress(len(batch_refs))
    return translations


def write_translation(page: fitz.Page, rect: fitz.Rect, translation: str) -> None:
    rect = fitz.Rect(rect.x0 - 0.8, rect.y0 - 0.6, rect.x1 + 0.8, rect.y1 + 0.6)
    page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
    # Paragraph rectangles can use a readable font first; shrink only when necessary.
    for size in (13.0 - step * 0.5 for step in range(16)):
        if page.insert_textbox(
            rect, translation, fontsize=size, fontname="china-s",
            color=(0, 0, 0), lineheight=1.13, overlay=True,
        ) >= 0:
            return
    page.insert_textbox(rect, translation, fontsize=5.5, fontname="china-s", color=(0, 0, 0), overlay=True)


def process_document(content: bytes, filename: str, language: str) -> tuple[bytes, str]:
    set_progress("extracting", "正在读取 PDF", completed=0, total=0)
    try:
        source = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise HTTPException(400, "This PDF could not be opened. Password-protected files are not supported.") from exc

    try:
        if len(source) == 0:
            raise HTTPException(400, "This PDF has no pages.")
        if len(source) > MAX_PAGES:
            raise HTTPException(400, f"This local app is limited to {MAX_PAGES} pages. Change MAX_PAGES in .env when ready.")

        page_blocks = [paragraph_blocks(page) for page in source]
        if not any(page_blocks):
            raise HTTPException(400, "No selectable text was found. This appears to be a scanned/image-only PDF; OCR is not enabled yet.")

        client = openai_client()
        translated_blocks = translate_document(client, page_blocks, language)

        set_progress("building", "正在重建译后 PDF", completed=0, total=len(source), pages=len(source))
        output = fitz.open()
        try:
            for page_no, original in enumerate(source):
                new = output.new_page(width=original.rect.width, height=original.rect.height)
                new.show_pdf_page(original.rect, source, page_no)
                for block_no, block in enumerate(page_blocks[page_no]):
                    write_translation(new, block.rect, translated_blocks[(page_no, block_no)])
                set_progress("building", "正在重建译后 PDF", completed=page_no + 1, total=len(source), pages=len(source))
            data = output.tobytes(garbage=4, deflate=True)
        finally:
            output.close()
    finally:
        source.close()

    base = os.path.splitext(filename)[0]
    set_progress("ready", "译后 PDF 已完成", completed=1, total=1)
    return data, base


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/progress")
def progress():
    with _progress_lock:
        return dict(_progress)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "max_pages": MAX_PAGES,
    }


@app.post("/translate")
async def convert(file: UploadFile = File(...), language: str = Form("Chinese (Simplified)")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    try:
        content = await file.read()
        data, base = await asyncio.to_thread(process_document, content, file.filename, language)
    except Exception:
        set_progress("error", "翻译未完成")
        raise

    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{base}_translated.pdf"'},
    )
