import os
import json
import litserve as ls
from transformers import pipeline

QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")


class QwenAPI(ls.LitAPI):
    def setup(self, device):
        print(f"[QwenAPI] Loading {QWEN_MODEL} …")
        self._pipe = pipeline(
            "text-generation",
            model=QWEN_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
        print(f"[QwenAPI] {QWEN_MODEL} ready.")

    def decode_request(self, request):
        return {
            "messages": [
                {
                    "role": "system",
                    "content": request.get("system", "")
                    + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no extra text.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        request.get("user_payload", {}), ensure_ascii=False
                    ),
                },
            ],
            "temperature": float(request.get("temperature", 0.1)),
            "max_tokens": int(request.get("max_tokens", 800)),
        }

    def predict(self, data):
        temp = max(float(data["temperature"]), 0.01)
        outputs = self._pipe(
            data["messages"],
            max_new_tokens=data["max_tokens"],
            temperature=temp,
            do_sample=temp > 0.01,
            return_full_text=False,
        )
        raw = outputs[0]["generated_text"]
        if isinstance(raw, list):
            raw = raw[-1].get("content", "")
        return {"text": str(raw)}

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    server = ls.LitServer(QwenAPI(), accelerator="auto", workers_per_device=1)
    server.run(port=8000)
