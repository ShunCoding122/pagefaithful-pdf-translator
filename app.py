"""Local-first PDF ebook translator.

The web page and API run only on http://127.0.0.1. Uploaded files are held in
memory and are never written to disk by this application.
"""

import io
import json
import os
import re
from dataclasses import dataclass

import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI

APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "300"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
app = FastAPI(title="PageFaithful PDF Translator")


@dataclass
class TextBlock:
    rect: fitz.Rect
    text: str


def openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY is not configured. Create .env from .env.example, then restart the app.")
    return OpenAI()


def text_blocks(page: fitz.Page) -> list[TextBlock]:
    blocks = []
    for item in page.get_text("dict")["blocks"]:
        if item["type"] != 0:
            continue
        text = "\n".join("".join(span["text"] for span in line["spans"]) for line in item["lines"]).strip()
        if len(text) >= 2:
            blocks.append(TextBlock(fitz.Rect(item["bbox"]), text))
    return blocks


def translate_one(client: OpenAI, text: str, language: str) -> str:
    prompt = (
        f"Translate the following private ebook text into {language}. Preserve paragraph breaks, "
        "headings, numbers, names, citations, and punctuation. Return only the translation.\n\n"
        + text
    )
    return client.responses.create(model=MODEL, input=prompt, store=False).output_text.strip()


def translate_batch(client: OpenAI, texts: list[str], language: str) -> list[str]:
    # Structured Outputs prevents malformed JSON. If the model still omits a block,
    # retry only that block, so one bad fragment never fails the whole book.
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
        f"Translate each item's text into {language}. This is text from a privately owned ebook. "
        "Preserve paragraph breaks, headings, numbers, names, citations, and inline punctuation. "
        "Return every supplied id exactly once.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        text={"format": {"type": "json_schema", "name": "block_translations", "strict": True, "schema": schema}},
        store=False,
    )
    try:
        items = json.loads(response.output_text)["translations"]
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


def translate_page(client: OpenAI, blocks: list[TextBlock], language: str) -> list[str]:
    translations, batch, chars = [], [], 0
    for block in blocks:
        if batch and chars + len(block.text) > 10_000:
            translations.extend(translate_batch(client, batch, language))
            batch, chars = [], 0
        batch.append(block.text)
        chars += len(block.text)
    if batch:
        translations.extend(translate_batch(client, batch, language))
    return translations


def write_translation(page: fitz.Page, rect: fitz.Rect, translation: str) -> None:
    # MuPDF bundles this CJK font: no extra Chinese-font setup is required.
    rect = fitz.Rect(rect.x0 - 0.8, rect.y0 - 0.5, rect.x1 + 0.8, rect.y1 + 0.5)
    page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
    start = min(13.0, max(6.5, rect.height / max(1.8, len(translation) / 22)))
    for size in (start - i * 0.5 for i in range(18)):
        if size < 4.5:
            break
        if page.insert_textbox(rect, translation, fontsize=size, fontname="china-s", color=(0, 0, 0), lineheight=1.12, overlay=True) >= 0:
            return
    page.insert_textbox(rect, translation, fontsize=4.5, fontname="china-s", color=(0, 0, 0), overlay=True)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/translate")
async def convert(file: UploadFile = File(...), language: str = Form("Chinese (Simplified)")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    try:
        source = fitz.open(stream=await file.read(), filetype="pdf")
    except Exception as exc:
        raise HTTPException(400, "This PDF could not be opened. Password-protected files are not supported.") from exc
    if len(source) == 0:
        raise HTTPException(400, "This PDF has no pages.")
    if len(source) > MAX_PAGES:
        raise HTTPException(400, f"This local test is limited to {MAX_PAGES} pages. Change MAX_PAGES in .env when ready.")

    page_blocks = [text_blocks(page) for page in source]
    if not any(page_blocks):
        raise HTTPException(400, "No selectable text was found. This appears to be a scanned/image-only PDF; OCR is the next feature to add.")

    client, output = openai_client(), fitz.open()
    try:
        for page_no, original in enumerate(source):
            new = output.new_page(width=original.rect.width, height=original.rect.height)
            new.show_pdf_page(original.rect, source, page_no)
            blocks = page_blocks[page_no]
            translations = translate_page(client, blocks, language) if blocks else []
            for block, translation in zip(blocks, translations):
                write_translation(new, block.rect, translation)
        data = output.tobytes(garbage=4, deflate=True)
    finally:
        output.close()
        source.close()

    base = os.path.splitext(file.filename)[0]
    return StreamingResponse(io.BytesIO(data), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{base}_translated.pdf"'})
