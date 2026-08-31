# PageFaithful — local-first PDF translator

Translate a text-based ebook PDF on your own computer while retaining each page, image, and approximate text-block position. The browser UI is local: `http://127.0.0.1:8000`.

## What it does now

- Works with English, French, German, Japanese, and other **selectable-text** PDFs.
- Sends only extracted text blocks to the translation API; it does not upload PDF files itself.
- Keeps PDF processing and the resulting file on this computer.
- Uses an OpenAI API key **only on this computer** through a local `.env` file.
- Allows up to 300 pages per run, groups visual lines into paragraphs, and shows live progress while translating., so you can judge quality and cost before translating a whole book.

## First run on Windows

1. Install [Python 3.11+](https://www.python.org/downloads/) and tick **Add Python to PATH** during setup.
2. Open PowerShell in this folder and run:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   notepad .env
   ```

3. In `.env`, paste the API key you create at [OpenAI Platform](https://platform.openai.com/api-keys):

   ```env
   OPENAI_API_KEY=sk-your-key-goes-here
   OPENAI_MODEL=gpt-4.1-mini
   MAX_PAGES=300
   TRANSLATION_WORKERS=4
   BATCH_CHAR_LIMIT=12000
   ```

   The key stays only in `.env`. Never commit it to GitHub or paste it into the webpage.

4. Start the local service:

   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```

5. Open http://127.0.0.1:8000 and translate a short, text-based PDF first.

Press `Ctrl+C` in PowerShell to stop it.

## Scope and next steps

This local version groups normal ebook text into paragraphs before translating it, leaves image blocks untouched, and does not bypass DRM, work on password-protected files, or accept image-only/scanned PDFs. OCR and cloud deployment can be added after the page-layout result is validated. For cloud deployment, the same application will use a server-side secret rather than the `.env` file.
