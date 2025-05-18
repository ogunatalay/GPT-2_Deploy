!pip install -q fastapi uvicorn transformers torch pyngrok nest-asyncio

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from pyngrok import ngrok
import nest_asyncio
import threading
from typing import Optional
import time
from functools import lru_cache

nest_asyncio.apply()

# Model caching decorator
def cache_model(maxsize=1):
    return lru_cache(maxsize=maxsize)

@cache_model()
def load_model(model_name: str):
    print(f"Loading model {model_name}...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return model

@cache_model()
def load_tokenizer(model_name: str):
    print(f"Loading tokenizer {model_name}...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    return tokenizer

app = FastAPI(title="Optimized Text Generation API", docs_url="/docs", redoc_url=None)

class PromptRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-load model and tokenizer at startup
print("Pre-loading model and tokenizer...")
start_load_time = time.time()
tokenizer = load_tokenizer(MODEL_NAME)
model = load_model(MODEL_NAME)
print(f"Total pre-load time: {time.time() - start_load_time:.2f} seconds")

@app.post("/generate", response_model=dict)
async def generate_text(request: PromptRequest, fastapi_request: Request):
    start_time = time.time()

    # Measure tokenization time
    tokenize_start = time.time()
    inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
    tokenize_time = time.time() - tokenize_start

    # Measure generation time
    generate_start = time.time()
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
    generate_time = time.time() - generate_start

    # Measure decoding time
    decode_start = time.time()
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decode_time = time.time() - decode_start

    generated_text = full_text[len(request.prompt):].strip()
    total_time = time.time() - start_time

    # Print latency metrics
    print(f"\nRequest latency breakdown:")
    print(f"Tokenization: {tokenize_time:.4f}s")
    print(f"Generation: {generate_time:.4f}s")
    print(f"Decoding: {decode_time:.4f}s")
    print(f"Total API time: {total_time:.4f}s")

    return {
        "generated_text": generated_text,
        "original_prompt": request.prompt,
        "parameters": request.dict(exclude={"prompt"}),
        "metrics": {
            "total_time": total_time,
            "tokenization_time": tokenize_time,
            "generation_time": generate_time,
            "decoding_time": decode_time
        }
    }

@app.get("/test")
async def test_endpoint():
    """Manual test endpoint with sample prompts"""
    test_prompts = [
        "The weather is beautiful today",
        "Artificial intelligence will in the future",
        "The food at this restaurant"
    ]

    results = []
    for prompt in test_prompts:
        start_time = time.time()
        request = PromptRequest(prompt=prompt, max_length=50)
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
        elapsed = time.time() - start_time

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "response_time": elapsed
        })

    return {"test_results": results}

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=False)

def setup_ngrok():
    NGROK_AUTH_TOKEN = "YOUR_TOKEN"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    tunnel = ngrok.connect(8000, bind_tls=True)
    print(f"\nNgrok tunnel: {tunnel.public_url}")
    print(f"API docs: {tunnel.public_url}/docs")
    print(f"Test endpoint: {tunnel.public_url}/test\n")
    return tunnel

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    setup_ngrok()

    server_thread.join()
