import os
import json
import litserve as ls
from litgpt import LLM

QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Qwen chat template tokens
_SYS_OPEN  = "<|im_start|>system\n"
_SYS_CLOSE = "<|im_end|>\n"
_USR_OPEN  = "<|im_start|>user\n"
_USR_CLOSE = "<|im_end|>\n"
_ASS_OPEN  = "<|im_start|>assistant\n"


def _build_prompt(system: str, user: str) -> str:
    return f"{_SYS_OPEN}{system}{_SYS_CLOSE}{_USR_OPEN}{user}{_USR_CLOSE}{_ASS_OPEN}"


class QwenAPI(ls.LitAPI):
    def setup(self, device):
        print(f"[QwenAPI] Loading {QWEN_MODEL} on {device} …")
        self.llm = LLM.load(QWEN_MODEL, accelerator=device)
        print(f"[QwenAPI] {QWEN_MODEL} ready.")

    def decode_request(self, request):
        # request: {system, user_payload, temperature, max_tokens}
        return {
            "prompt": _build_prompt(
                system=request.get("system", ""),
                user=json.dumps(request.get("user_payload", {}), ensure_ascii=False),
            ),
            "temperature": float(request.get("temperature", 0.1)),
            "max_tokens":  int(request.get("max_tokens", 800)),
        }

    def predict(self, data):
        return self.llm.generate(
            data["prompt"],
            max_new_tokens=data["max_tokens"],
            temperature=max(data["temperature"], 0.01),
            top_k=50,
        )

    def encode_response(self, output):
        return {"text": output}


if __name__ == "__main__":
    server = ls.LitServer(QwenAPI(), accelerator="auto", workers_per_device=1)
    server.run(port=8000)
