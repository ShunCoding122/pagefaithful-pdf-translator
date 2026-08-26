import io, os, json
import fitz
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI

app=FastAPI(title='PageFaithful PDF Translator')
client=OpenAI()

def translate(text, language):
    if not text.strip(): return text
    r=client.responses.create(model=os.getenv('OPENAI_MODEL','gpt-5.6'), input=f'Translate the following into {language}. Preserve paragraph breaks, headings, numbers, names, citations and inline emphasis. Return ONLY the translation.\n\n{text}')
    return r.output_text.strip()

@app.post('/translate')
async def convert(file: UploadFile=File(...), language: str=Form('Chinese (Simplified)')):
    if not file.filename.lower().endswith('.pdf'): raise HTTPException(400,'Please upload a PDF.')
    src=fitz.open(stream=await file.read(), filetype='pdf'); out=fitz.open()
    for page in src:
        new=out.new_page(width=page.rect.width,height=page.rect.height)
        new.show_pdf_page(page.rect,src,page.number) # keeps artwork and original page geometry
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0: continue
            rect=fitz.Rect(block['bbox']); text='\n'.join(''.join(s['text'] for s in l['spans']) for l in block['lines'])
            if not text.strip(): continue
            # cover original text then typeset translated text inside its original region
            new.draw_rect(rect, color=None, fill=(1,1,1), overlay=True)
            translated=translate(text,language)
            size=max(5,min(11,rect.height/max(1, len(translated)/28)))
            new.insert_textbox(rect,translated,fontsize=size,fontname='helv',color=(0,0,0),overlay=True)
    data=out.tobytes(garbage=4,deflate=True)
    return StreamingResponse(io.BytesIO(data),media_type='application/pdf',headers={'Content-Disposition':'attachment; filename=translated.pdf'})
