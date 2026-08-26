# PageFaithful PDF Translator

`OPENAI_API_KEY=... uvicorn app:app --reload` then open `http://127.0.0.1:8000/static/index.html`.

For deployment, add `OPENAI_API_KEY` as a server secret; never put it in browser code. This MVP preserves original pages and replaces extractable text blocks in place. It is intended for PDF ebooks you own and may translate for personal use.
