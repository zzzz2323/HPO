import numpy as np
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from global_llm import GlobalLLM, glm_chat
import json
import re


# ============================================================
# Baseline embedder
# ============================================================

class BaselineEmbedder:
    def __init__(self, model_path: str = "/public/home/202230275320/medicoding/models/bge"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)

    def encode(self, text: str) -> np.ndarray:
        emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb.reshape(1, -1)

    def similarity(self, a: str, b: str) -> float:
        va = self.encode(a)
        vb = self.encode(b)
        return float(np.dot(va, vb.T).item())


# ============================================================
# Local LLM — 让模型“直接输出候选行”
# ============================================================

class LocalLLM:
    def __init__(self, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            pack = GlobalLLM.get()   # 单例 LLM
            self.model = pack["model"]
            self.tokenizer = pack["tokenizer"]
        else:
            self.model = model
            self.tokenizer = tokenizer

    def _extract_hp_id_from_text(self, text: str) -> str | None:
        """
        从 LLM 输出中抓取第一个 HP:XXXXXXX
        """
        m = re.search(r"HP:\d{7}", text)
        if m:
            return m.group(0)
        return None

    def choose_hpo(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        不再用 JSON/choice 编号，让模型直接输出候选中的一行，
        然后从中抽取 HP ID。
        """

        mention = prompt["mention"]
        context = prompt["context"]
        candidates = prompt["candidates"][:10]  # 最多 10 个候选

        cand_list = "\n".join(
            f"{i+1}. {c['id']} {c['label']}"
            for i, c in enumerate(candidates)
        )
        cand_ids = {c["id"] for c in candidates}

        instruction = f"""
You are a medical terminology assistant.

Mention:
{mention}

Sentence context:
{context}

Candidates (do not change them):
{cand_list}

Your task:
Select the *single best matching* candidate from the list.

Important rules:
- Your output must be *exactly one entire line* copied from the candidates list.
- Do NOT add explanation.
- Do NOT add extra words.
- Do NOT output JSON.
- Do NOT output multiple lines.
- Output must match *one and only one* of the candidate lines exactly.

Now output the best candidate:
"""

        for attempt in range(4):
            try:
                text = glm_chat(
                    self.model,
                    self.tokenizer,
                    instruction,
                    max_new_tokens=64
                )
                # 调试时可以保留这句
                print(text)

                # 取第一行非空行
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                if not lines:
                    continue
                first = lines[0]

                hp_id = self._extract_hp_id_from_text(first)
                if hp_id is None:
                    # 再从整个输出里扫一遍
                    hp_id = self._extract_hp_id_from_text(text)

                if hp_id and hp_id in cand_ids:
                    return {
                        "id": hp_id,
                        "reason": "llm_selected_line",
                        "confidence": 0.9
                    }

            except Exception as e:
                if attempt == 3:
                    print(f"[LLM TOTAL FAIL] {e} | mention='{mention}'")

        # 完全失败：回退到第一个候选
        return {
            "id": candidates[0]["id"],
            "reason": "llm_emergency_fallback",
            "confidence": 0.5
        }


# ============================================================
# Normalizer
# ============================================================

class Normalizer:
    def __init__(self,
                 embedder=None,
                 llm=None,
                 llm_threshold=0.06,
                 score_threshold=0.8,
                 enable_llm_normalize=True): # [修改] 增加了控制参数，默认为 True

        self.embedder = embedder or BaselineEmbedder()
        self.llm = llm or LocalLLM()

        self.llm_threshold = llm_threshold
        self.score_threshold = score_threshold
        self.enable_llm_normalize = enable_llm_normalize  # [修改] 保存该参数

        self.stats = {
            "total": 0,
            "baseline_confident": 0,
            "llm_triggered": 0,
            "llm_helped": 0,
            "trace": []
        }

    # -------------------------
    # Baseline scoring
    # -------------------------
    def _baseline_score(self, mention_text, sent_context, candidate):
        query = f"{mention_text} ; {sent_context}"
        return self.embedder.similarity(query, candidate["label"])

    # -------------------------
    # Rank candidates
    # -------------------------
    def _rank_candidates(self, mention, context, candidates):
        if not candidates:
            return []

        query_text = f"{mention} ; {context}"
        query_vec = self.embedder.encode(query_text)

        labels = [c["label"] for c in candidates]
        cand_vecs = self.embedder.model.encode(
            labels,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        sims = cand_vecs @ query_vec.T

        ranked = []
        for i, c in enumerate(candidates):
            s = float(sims[i, 0])
            ranked.append({
                "id": c["id"],
                "label": c["label"],
                "score": s,
                "raw": c
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    # -------------------------
    # LLM trigger logic
    # -------------------------
    def _should_trigger_llm(self, ranked):
        if len(ranked) < 2:
            return False

        top1, top2 = ranked[0], ranked[1]
        return (
            abs(top1["score"] - top2["score"]) < self.llm_threshold or
            top1["score"] < self.score_threshold
        )

    # -------------------------
    # LLM choose
    # -------------------------
    def _llm_choose(self, mention, context, ranked):
        options = [{"id": c["id"], "label": c["label"]} for c in ranked]

        prompt = {
            "mention": mention,
            "context": context,
            "candidates": options
        }

        res = self.llm.choose_hpo(prompt)
        cid = res["id"]

        if cid not in [c["id"] for c in ranked]:
            cid = ranked[0]["id"]
            method = "llm_invalid_fallback"
        else:
            method = "llm"
            if cid != ranked[0]["id"]:
                self.stats["llm_helped"] += 1

        return {
            "id": cid,
            "method": method,
            "confidence": res.get("confidence", ranked[0]["score"]),
            "reason": res.get("reason", "llm_selected_line")
        }

    # -------------------------
    # Main entry
    # -------------------------
    def normalize(self, mention_text, sent_context, candidates):
        self.stats["total"] += 1

        if not candidates:
            result = {
                "id": None,
                "method": "no_candidate",
                "confidence": 0.0,
                "reason": "no_candidate",
                "ranked": []
            }
            self.stats["trace"].append(result)
            return result

        ranked = self._rank_candidates(mention_text, sent_context, candidates)
        if not ranked:
            result = {
                "id": None,
                "method": "no_ranked",
                "confidence": 0.0,
                "reason": "no_ranked",
                "ranked": []
            }
            self.stats["trace"].append(result)
            return result

        # [修改] 核心逻辑修改：
        # 只有当 enable_llm_normalize 为 True，并且 _should_trigger_llm 返回 True 时，
        # 才进入 else 分支调用 LLM。
        # 否则（如果 switch 为 False 或 trigger 为 False），都直接走 baseline。
        if self.enable_llm_normalize and self._should_trigger_llm(ranked):
            self.stats["llm_triggered"] += 1
            result = self._llm_choose(mention_text, sent_context, ranked)
        else:
            self.stats["baseline_confident"] += 1
            result = {
                "id": ranked[0]["id"],
                "method": "baseline",
                "confidence": ranked[0]["score"],
                "reason": "baseline_confident",
                "ranked": ranked
            }

        result["ranked"] = ranked
        self.stats["trace"].append(result)
        return result