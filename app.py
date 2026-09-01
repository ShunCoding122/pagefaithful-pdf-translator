"""Local-first, layout-aware PDF ebook translator."""

import asyncio
import io
import json
import os
import re
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
FONT_NAME = "pagefaithful_cjk"
BODY_FONT_SIZE = float(os.getenv("BODY_FONT_SIZE", "10.5"))
BODY_LINE_HEIGHT = float(os.getenv("BODY_LINE_HEIGHT", "1.42"))
PAGE_MARGIN_X = float(os.getenv("PAGE_MARGIN_X", "54"))
PAGE_MARGIN_Y = float(os.getenv("PAGE_MARGIN_Y", "58"))
PARAGRAPH_GAP = float(os.getenv("PARAGRAPH_GAP", "8"))


def resolve_cjk_font() -> str | None:
    candidates = [
        os.getenv("PDF_TRANSLATOR_FONT", ""),
        r"C:\\Windows\\Fonts\\msyh.ttc",   # Microsoft YaHei, standard on Chinese Windows
        r"C:\\Windows\\Fonts\\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


CJK_FONT_PATH = resolve_cjk_font()

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
        "numbers, quotations, citations, and paragraph meaning. Preserve paragraph breaks. Return only the translation.\n\n"
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
        "Keep every supplied id exactly once, preserving paragraph breaks within each text. Do not summarize or add commentary.\n\n"
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


def normalize_translation(text: str) -> str:
    """Remove layout artefacts while keeping spaces inside Latin words intact."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、】【）》〉」』])", r"\1", text)
    text = re.sub(r"([（【《〈「『])\s+", r"\1", text)
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[A-Za-z0-9])", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u3400-\u9fff])", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def reader_rect(page: fitz.Page, top: float | None = None) -> fitz.Rect:
    return fitz.Rect(PAGE_MARGIN_X, PAGE_MARGIN_Y if top is None else top,
                     page.rect.width - PAGE_MARGIN_X, page.rect.height - PAGE_MARGIN_Y)


def add_reader_page(output: fitz.Document, template: fitz.Page) -> fitz.Page:
    if not CJK_FONT_PATH:
        raise HTTPException(500, "No embeddable Chinese font was found. Set PDF_TRANSLATOR_FONT in .env to a Chinese .ttf or .ttc font file.")
    page = output.new_page(width=template.rect.width, height=template.rect.height)
    page.insert_font(fontname=FONT_NAME, fontfile=CJK_FONT_PATH)
    return page


def can_fit(measure_page: fitz.Page, rect: fitz.Rect, text: str) -> bool:
    return measure_page.insert_textbox(rect, text, fontsize=BODY_FONT_SIZE,
                                       fontname=FONT_NAME, lineheight=BODY_LINE_HEIGHT) >= 0


def sentence_fragments(text: str) -> list[str]:
    fragments = [part for part in re.split(r"(?<=[。！？；…])", text) if part]
    return fragments or [text]


def largest_fitting_prefix(measure_page: fitz.Page, rect: fitz.Rect, text: str) -> tuple[str, str]:
    """Prefer a sentence boundary when an unusually long paragraph must span pages."""
    prefix = ""
    for fragment in sentence_fragments(text):
        candidate = prefix + fragment
        if can_fit(measure_page, rect, candidate):
            prefix = candidate
        else:
            break
    if prefix:
        return prefix, text[len(prefix):]

    low, high, best = 1, len(text), 0
    while low <= high:
        midpoint = (low + high) // 2
        if can_fit(measure_page, rect, text[:midpoint]):
            best, low = midpoint, midpoint + 1
        else:
            high = midpoint - 1
    if best == 0:
        raise HTTPException(500, "The selected reader font is too large for this page size.")
    break_at = max(text.rfind("，", 0, best), text.rfind("、", 0, best), text.rfind(" ", 0, best))
    if break_at > max(1, best // 2):
        best = break_at + 1
    return text[:best], text[best:]


def render_reader_document(source: fitz.Document, page_blocks: list[list[TextBlock]],
                           translated_blocks: dict[tuple[int, int], str]) -> bytes:
    """Create a clean, reflowable reading PDF instead of forcing text into source pages."""
    if not CJK_FONT_PATH:
        raise HTTPException(500, "No embeddable Chinese font was found. Set PDF_TRANSLATOR_FONT in .env to a Chinese .ttf or .ttc font file.")
    output, measure_doc = fitz.open(), fitz.open()
    measure_page = measure_doc.new_page(width=source[0].rect.width, height=source[0].rect.height)
    measure_page.insert_font(fontname=FONT_NAME, fontfile=CJK_FONT_PATH)
    current: fitz.Page | None = None
    cursor = PAGE_MARGIN_Y

    def start_page(template: fitz.Page) -> fitz.Page:
        nonlocal current, cursor
        current, cursor = add_reader_page(output, template), PAGE_MARGIN_Y
        return current

    def place(text: str, template: fitz.Page) -> None:
        nonlocal current, cursor
        if current is None:
            start_page(template)
        remaining, first_piece = text, True
        while remaining:
            assert current is not None
            if cursor >= current.rect.height - PAGE_MARGIN_Y - 0.5:
                start_page(template)
            available = reader_rect(current, cursor)
            candidate = ("　　" if first_piece else "") + remaining
            if can_fit(measure_page, available, candidate):
                leftover = current.insert_textbox(available, candidate, fontsize=BODY_FONT_SIZE,
                    fontname=FONT_NAME, color=(0, 0, 0), lineheight=BODY_LINE_HEIGHT)
                cursor = available.y1 - leftover + PARAGRAPH_GAP
                return
            if can_fit(measure_page, reader_rect(current), candidate):
                start_page(template)
                continue
            prefix, remaining = largest_fitting_prefix(measure_page, available, candidate)
            leftover = current.insert_textbox(available, prefix, fontsize=BODY_FONT_SIZE,
                fontname=FONT_NAME, color=(0, 0, 0), lineheight=BODY_LINE_HEIGHT)
            cursor = available.y1 - leftover + PARAGRAPH_GAP
            first_piece = False
            if remaining:
                start_page(template)

    try:
        for page_no, original in enumerate(source):
            blocks = page_blocks[page_no]
            if not blocks:
                if original.get_images(full=True):
                    image_page = output.new_page(width=original.rect.width, height=original.rect.height)
                    image_page.show_pdf_page(original.rect, source, page_no)
                    current = None
                continue
            for block_no, _block in enumerate(blocks):
                translated = normalize_translation(translated_blocks[(page_no, block_no)])
                for paragraph in (part.strip() for part in re.split(r"\n\s*\n+", translated)):
                    if paragraph:
                        place(paragraph, original)
        return output.tobytes(garbage=4, deflate=True)
    finally:
        measure_doc.close()
        output.close()


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

        set_progress("building", "正在排版为连续阅读版 PDF", completed=0, total=len(source), pages=len(source))
        data = render_reader_document(source, page_blocks, translated_blocks)
        set_progress("building", "正在排版为连续阅读版 PDF", completed=len(source), total=len(source), pages=len(source))
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
