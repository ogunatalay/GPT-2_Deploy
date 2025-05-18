# locustfile.py
from locust import HttpUser, task, between
import random

sample_prompts = [
    "Türkiye'nin başkenti",
    "Yapay zeka nedir?",
    "Python programlama dili",
    "Gelecekte teknoloji",
    "İstanbul'un tarihi"
]


class ApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(1)
    def test_sync_endpoint(self):
        prompt = random.choice(sample_prompts)
        self.client.post("/generate_sync", json={
            "prompt": prompt,
            "max_length": 30
        })

    @task(1)
    def test_async_endpoint(self):
        prompt = random.choice(sample_prompts)
        self.client.post("/generate_async", json={
            "prompt": prompt,
            "max_length": 30
        })