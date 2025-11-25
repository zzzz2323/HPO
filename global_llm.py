# global_llm.py
# 用于全局只加载一次本地 ChatGLM（ZhipuAI/glm-edge-1.5b-chat 或 6B）

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class GlobalLLM:
    """
    全局单例 LLM 加载器：
    - 在第一次调用 get() 时加载模型
    - 后续所有模块（C 阶段 Normalizer / D 阶段 Assertion / E 阶段 Modifier）
      都复用同一个模型实例，避免重复加载 GPU 权重
    """

    _tokenizer = None
    _model = None

    @staticmethod
    def load(model_path: str,
             dtype: torch.dtype = torch.float16,
             device: Optional[str] = None):
        """
        显式加载模型（可选）
        """
        if GlobalLLM._model is None:
            print(f"[GlobalLLM] Loading model from: {model_path} ...")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model = model.to(device).eval()
            if device == "cuda":
                model = model.half()

            GlobalLLM._tokenizer = tokenizer
            GlobalLLM._model = model

            print("[GlobalLLM] Loaded successfully.")

    @staticmethod
    def get(model_path: str = "/public/home/202230275320/medicoding/models/glm-1.5B"):
        """
        自动加载 + 返回全局单例
        """
        if GlobalLLM._model is None:
            GlobalLLM.load(model_path)
        return {
            "model": GlobalLLM._model,
            "tokenizer": GlobalLLM._tokenizer
        }


# ============================================================
# 统一 GLM chat wrapper（必须有的函数）
# ============================================================

def glm_chat(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """
    尝试使用 model.chat()；若不存在 .chat 方法则自动 fallback 到 generate()
    """
    # 如果模型有 chat 方法
    if hasattr(model, "chat"):
        try:
            resp = model.chat(
                tokenizer,
                prompt,
                history=[],
                temperature=0.8,
                top_p=0.95,
                max_length=max_new_tokens
            )
            print(resp)
            if isinstance(resp, tuple):
                text, _ = resp
            else:
                text = resp
            return text.strip()
        except Exception:
            pass  # 失败 fallback

    # fallback 使用 generate
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip()
