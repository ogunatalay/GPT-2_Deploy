!pip install -q fastapi uvicorn transformers torch pyngrok nest-asyncio


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from pyngrok import ngrok
import nest_asyncio
import threading
from typing import Optional

nest_asyncio.apply()

app = FastAPI(title="Metin Oluşturma API", docs_url="/docs", redoc_url=None)

class PromptRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

@app.post("/generate", response_model=dict)
async def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = full_text[len(request.prompt):].strip()
    return {
        "generated_text": generated_text,
        "original_prompt": request.prompt,
        "parameters": request.dict(exclude={"prompt"})
    }

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=False)

def setup_ngrok():
    NGROK_AUTH_TOKEN = "YOUR TOKEN"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    tunnel = ngrok.connect(8000, bind_tls=True)
    print(f"Ngrok tüneli: {tunnel.public_url}")
    print(f"API docs: {tunnel.public_url}/docs")
    return tunnel

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    setup_ngrok()

    server_thread.join()

