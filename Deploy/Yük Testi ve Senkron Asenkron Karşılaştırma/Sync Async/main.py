# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import pandas as pd
from typing import Optional
import asyncio

app = FastAPI(title="Metin Oluşturma API - Performans Testi", docs_url="/docs")

MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model ve tokenizer'ı yükle (global olarak)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)


class PromptRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 50
    temperature: Optional[float] = 0.7


# Senkron (sync) versiyon
@app.post("/generate_sync")
def generate_text_sync(request: PromptRequest):
    start_time = time.time()

    inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    latency = time.time() - start_time
    return {
        "generated_text": generated_text,
        "latency": latency,
        "endpoint_type": "sync"
    }


# Asenkron (async) versiyon
@app.post("/generate_async")
async def generate_text_async(request: PromptRequest):
    start_time = time.time()

    # CPU-bound işlemleri thread pool'da çalıştır
    loop = asyncio.get_event_loop()

    # Tokenization ve generation işlemlerini thread'de çalıştır
    def _generate():
        inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated_text = await loop.run_in_executor(None, _generate)

    latency = time.time() - start_time
    return {
        "generated_text": generated_text,
        "latency": latency,
        "endpoint_type": "async"
    }


# Metrikleri kaydetmek için
performance_data = []


@app.get("/performance_data")
async def get_performance_data():
    df = pd.DataFrame(performance_data)

    # Sonuçları CSV olarak kaydet
    df.to_csv("performance_results.csv", index=False)

    # Ortalama değerleri hesapla
    avg_results = {
        "avg_latency_sync": df[df['endpoint_type'] == 'sync']['latency'].mean(),
        "avg_latency_async": df[df['endpoint_type'] == 'async']['latency'].mean(),
        "total_requests": len(df)
    }

    return {
        "raw_data": "performance_results.csv olarak kaydedildi",
        "summary": avg_results
    }