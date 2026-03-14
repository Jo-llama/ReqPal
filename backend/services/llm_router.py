import os
import re
import json
import asyncio
import threading
from typing import Any, Dict, Tuple, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv(".env")


class LLMRateLimitError(Exception):
    pass


class LLMHTTP:
    """
    OpenAI-compatible provider (Groq/OpenAI)
    Accepts either:
      - base_url = https://api.openai.com/v1
      - OR base_url = https://api.openai.com/v1/chat/completions
    """
    def __init__(self, api_key: str, base_url: str, model: str, name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name

    def _chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def chat_json(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_s: float = 30.0,
        retries: int = 1,
        backoff_s: float = 0.8,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }

        url = self._chat_url()
        last_text = None

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    r = await client.post(url, headers=headers, json=payload)
            except httpx.ReadTimeout:
                if attempt >= retries:
                    raise
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            last_text = r.text

            if r.status_code == 429:
                if attempt >= retries:
                    raise LLMRateLimitError(f"{self.name} 429: {r.text}")
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            if 500 <= r.status_code < 600:
                if attempt >= retries:
                    r.raise_for_status()
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}

        raise RuntimeError(f"{self.name} call failed. last={last_text}")



class OllamaHTTP:
    """
    Native Ollama API:
      POST http://127.0.0.1:11434/api/chat
    """

    def __init__(self, base_url: str, model: str, name: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name

    def _ollama_options(self, temperature: float, max_tokens: int) -> Dict[str, Any]:
        # lighter defaults
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX") or 2048)
        max_predict_env = int(os.getenv("OLLAMA_MAX_PREDICT") or 384)

        num_predict = min(int(max_tokens), max_predict_env)

        return {
            "temperature": float(temperature),
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }

    async def chat_json(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 384,
        timeout_s: float = 180.0,   # longer for local
        retries: int = 1,
        backoff_s: float = 1.0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "options": self._ollama_options(temperature=temperature, max_tokens=max_tokens),
        }

        last_text = None

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    r = await client.post(url, json=payload)
            except httpx.ReadTimeout:
                if attempt >= retries:
                    raise
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            last_text = r.text

            if r.status_code == 429:
                if attempt >= retries:
                    raise LLMRateLimitError(f"{self.name} 429: {r.text}")
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            if 500 <= r.status_code < 600:
                if attempt >= retries:
                    r.raise_for_status()
                await asyncio.sleep(backoff_s * (2 ** attempt))
                continue

            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content") or ""

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}

        raise RuntimeError(f"{self.name} call failed. last={last_text}")


class QwenLocal:
    """
    Local Qwen model via HuggingFace transformers.
    Model is lazy-loaded on first call (or eagerly via warmup()).
    Set QWEN_MODEL env var to override (default: Qwen/Qwen2.5-7B-Instruct).
    Set QWEN_LOAD_4BIT=1 for 4-bit quantization (~5 GB VRAM instead of ~15 GB).
    """

    def __init__(self, model_id: str, name: str = "qwen"):
        self.model_id = model_id
        self.name = name
        self.model = model_id  # for providers_status()
        self._pipe = None
        self._lock = threading.Lock()

    def warmup(self):
        """Pre-load model at startup to avoid cold-start on first request."""
        self._load()

    def _load(self):
        if self._pipe is not None:
            return
        with self._lock:
            if self._pipe is not None:
                return
            print(f"[QwenLocal] Loading {self.model_id} …")
            import torch
            from transformers import pipeline
            kwargs: Dict[str, Any] = {
                "model": self.model_id,
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            if os.getenv("QWEN_LOAD_4BIT", "").strip() == "1":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            self._pipe = pipeline("text-generation", **kwargs)
            print(f"[QwenLocal] {self.model_id} ready.")

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {"raw": text}

    def _generate_sync(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        self._load()
        assert self._pipe is not None
        messages = [
            {
                "role": "system",
                "content": system + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no extra text.",
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]
        outputs = self._pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=max(float(temperature), 0.01),
            do_sample=float(temperature) > 0.01,
            return_full_text=False,
        )
        raw = outputs[0]["generated_text"]
        if isinstance(raw, list):
            raw = raw[-1].get("content", "")
        return self._extract_json(str(raw))

    async def chat_json(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout_s: float = 180.0,
        **kwargs,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            system,
            user_payload,
            temperature,
            max_tokens,
        )


class LitServeClient:
    """
    Calls the Qwen LitServe server (server.py) running on localhost:8000.
    Set LITSERVE_URL env var to override (default: http://127.0.0.1:8000).
    Primary provider — falls back to QwenLocal if server is not reachable.
    """

    def __init__(self, base_url: str, name: str = "litserve"):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.model = "qwen-litserve"

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {"raw": text}

    async def chat_json(
        self,
        system: str,
        user_payload: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout_s: float = 120.0,
        **kwargs,
    ) -> Dict[str, Any]:
        payload = {
            "input": {
                "system": system + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no extra text.",
                "user_payload": user_payload,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        }
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(f"{self.base_url}/predict", json=payload)
            r.raise_for_status()
            data = r.json()
            text = data.get("output", {}).get("text") or data.get("text") or ""
            return self._extract_json(text)


class LLMRouter:
    """
    Provider order:
      1) LitServe (server.py, port 8000)              <-- primary (Lightning AI)
      2) Qwen local via transformers                  <-- fallback if server not running
      3) OpenAI (if OPENAI_API_KEY set)
      4) Ollama (if OLLAMA_MODEL set)

    Returns: (json, provider_name, trace[])
    """

    def __init__(self):
        self.providers: List[Any] = []

        # 1. LitServe (primary on Lightning AI)
        litserve_url = (os.getenv("LITSERVE_URL") or "http://127.0.0.1:8000").strip()
        self.providers.append(LitServeClient(base_url=litserve_url, name="litserve"))

        # 2. Qwen local via transformers (fallback if server.py not running)
        qwen_model = (os.getenv("QWEN_MODEL") or "Qwen/Qwen2.5-7B-Instruct").strip()
        self.providers.append(QwenLocal(model_id=qwen_model, name="qwen"))

        # 2. OpenAI (optional fallback)
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if openai_key:
            self.providers.append(
                LLMHTTP(
                    api_key=openai_key,
                    base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/chat/completions").strip(),
                    model=(os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
                    name="openai",
                )
            )

        # 3. Ollama (optional fallback)
        ollama_url = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
        ollama_model = (os.getenv("OLLAMA_MODEL") or "").strip()
        if ollama_model:
            self.providers.append(OllamaHTTP(base_url=ollama_url, model=ollama_model, name="ollama"))

    def providers_status(self) -> Dict[str, Any]:
        return {
            "configured": [p.name for p in self.providers],
            "models": {p.name: p.model for p in self.providers},
        }

    async def chat_json(self, *args, **kwargs) -> Tuple[Dict[str, Any], str, List[str]]:
        trace: List[str] = []

        for p in self.providers:
            try:
                trace.append(f"try:{p.name}:{p.model}")
                out = await p.chat_json(*args, **kwargs)
                trace.append(f"ok:{p.name}")
                return out, p.name, trace
            except Exception as e:
                msg = str(e)
                if len(msg) > 200:
                    msg = msg[:200] + "…"
                trace.append(f"fail:{p.name}:{type(e).__name__}:{msg}")
                continue

        raise RuntimeError("All LLM providers failed. trace=" + " | ".join(trace))
