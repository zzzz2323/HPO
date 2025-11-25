from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re
import json

import torch
from transformers import AutoTokenizer, AutoModel

from global_llm import GlobalLLM


# ============================================================
# 1. 值域定义 + 同义词映射
# ============================================================

VALUE_DOMAINS: Dict[str, List[str]] = {
    "severity": ["mild", "moderate", "severe"],
    "laterality": ["left", "right", "bilateral", "unilateral"],
    "onset": ["infancy", "childhood", "adult", "neonatal", "congenital"],
    "frequency": ["recurrent", "occasional", "frequent", "intermittent"],
    "progression": ["progressive", "worsening", "improving", "stable", "deteriorating"],
}

VALUE_SYNONYMS: Dict[str, Dict[str, str]] = {
    "severity": {
        "mildly": "mild",
        "slight": "mild",
        "moderately": "moderate",
        "marked": "severe",
        "severely": "severe",
    },
    "laterality": {
        "left-sided": "left",
        "right-sided": "right",
        "both sides": "bilateral",
    },
    "onset": {
        "adult-onset": "adult",
        "since childhood": "childhood",
        "since infancy": "infancy",
    },
    "frequency": {
        "often": "frequent",
        "frequently": "frequent",
        "sporadic": "occasional",
    },
    "progression": {
        "getting worse": "worsening",
        "getting better": "improving",
        "unchanged": "stable",
    },
}


# ============================================================
# 2. 修饰抽取器（正则）
# ============================================================

@dataclass
class ModifierSpan:
    slot: str
    value: str
    raw_text: str
    char_start: int
    char_end: int


class ModifierExtractor:
    def __init__(self):
        self.patterns: Dict[str, re.Pattern] = {
            "severity": re.compile(
                r"\b(mild|mildly|slight|moderate|moderately|severe|severely|marked)\b",
                flags=re.IGNORECASE),
            "laterality": re.compile(
                r"\b(left|right|bilateral|unilateral|left-sided|right-sided|both sides)\b",
                flags=re.IGNORECASE),
            "onset": re.compile(
                r"\b(infancy|childhood|adult-onset|adult|neonatal|congenital|since childhood|since infancy)\b",
                flags=re.IGNORECASE),
            "frequency": re.compile(
                r"\b(recurrent|occasional|frequent|frequently|often|intermittent|sporadic)\b",
                flags=re.IGNORECASE),
            "progression": re.compile(
                r"\b(progressive|worsening|getting worse|improving|getting better|stable|unchanged|deteriorating)\b",
                flags=re.IGNORECASE),
        }

    def _canonicalize(self, slot: str, text: str) -> Optional[str]:
        t = text.lower()
        if t in VALUE_DOMAINS[slot]:
            return t
        syn = VALUE_SYNONYMS.get(slot, {})
        if t in syn:
            return syn[t]
        t_norm = t.replace("-", " ")
        for v in VALUE_DOMAINS[slot]:
            if v in t_norm:
                return v
        return None

    def extract(self, sentence: str) -> List[ModifierSpan]:
        spans = []
        for slot, pat in self.patterns.items():
            for m in pat.finditer(sentence):
                raw = m.group(0)
                canonical = self._canonicalize(slot, raw)
                if canonical:
                    spans.append(
                        ModifierSpan(
                            slot=slot,
                            value=canonical,
                            raw_text=raw,
                            char_start=m.start(),
                            char_end=m.end(),
                        )
                    )
        return spans


# ============================================================
# 3. LLM 
# ============================================================

class GlmEdgeModifierLLM:
    """
    使用本地 GLM（通过 GlobalLLM 单例）
    """

    def __init__(self, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            pack = GlobalLLM.get()
            self.model = pack["model"]
            self.tokenizer = pack["tokenizer"]
        else:
            self.model = model
            self.tokenizer = tokenizer

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        与 C / D 阶段一致的稳健 JSON 提取方法
        """
        t = text.strip()

        # 去除 markdown fence
        if t.startswith("```"):
            t = t.strip("`")
            idx = t.find("{")
            if idx != -1:
                t = t[idx:]

        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON found")
        return json.loads(t[start:end + 1])

    def _build_prompt(self, sentence, phenotypes, modifiers) -> str:
        ph_list = [
            {"index": i, "text": p.get("mention") or p.get("label")}
            for i, p in enumerate(phenotypes)
        ]

        mods_by_slot: Dict[str, List[str]] = {}
        for m in modifiers:
            mods_by_slot.setdefault(m.slot, [])
            if m.value not in mods_by_slot[m.slot]:
                mods_by_slot[m.slot].append(m.value)

        # 构造 JSON 模板：所有 phenotype indices 都出现
        # 小模型看到完整结构→复制→修改→稳定
        template = "{\n"
        for i in range(len(ph_list)):
            template += f'  "{i}": {{}},\n'
        template = template.rstrip(",\n") + "\n}"

        return (
            "You are a clinical NLP system for phenotype post-coordination.\n\n"
            "Your task:\n"
            "- Bind modifiers to phenotypes.\n"
            "- Use ONLY the modifiers provided.\n"
            "- At most one value per slot for each phenotype.\n"
            "- If no modifier applies, return an empty object {} for that phenotype.\n\n"
            "Output rules (very important):\n"
            "- Output must be EXACTLY one valid JSON object.\n"
            "- No explanation outside the JSON.\n"
            "- No comments, no markdown.\n"
            "- Do NOT repeat the input text.\n"
            "- Follow the structure of the template below.\n\n"
            "Sentence:\n"
            f"{sentence}\n\n"
            "Phenotypes:\n"
            f"{json.dumps(ph_list, ensure_ascii=False)}\n\n"
            "Available modifiers (grouped by slot):\n"
            f"{json.dumps(mods_by_slot, ensure_ascii=False)}\n\n"
            "JSON output template (copy this structure exactly):\n"
            f"{template}\n\n"
            "Now output ONLY the JSON object:"
        )


    def bind(self, sentence, phenotypes, modifiers):
        if not phenotypes or not modifiers:
            return {}

        prompt = self._build_prompt(sentence, phenotypes, modifiers)

        try:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[])
            parsed = self._extract_json(response)
        except Exception:
            # fallback 重解析失败→不绑定
            return {}

        result: Dict[int, Dict[str, str]] = {}
        for key, slotmap in parsed.items():
            try:
                idx = int(key)
            except:
                continue
            if idx < 0 or idx >= len(phenotypes):
                continue

            result[idx] = {}
            for slot, val in slotmap.items():
                v = str(val).lower()
                if slot in VALUE_DOMAINS and v in VALUE_DOMAINS[slot]:
                    result[idx][slot] = v

        return result


# ============================================================
# 4. PostCoordinator
# ============================================================

class PostCoordinator:
    def __init__(self,
                 extractor: Optional[ModifierExtractor] = None,
                 llm: Optional[GlmEdgeModifierLLM] = None,
                 ambiguity_threshold: int = 5,
                 enable_llm_post_coordination: bool = True): # [修改] 增加控制参数

        self.extractor = extractor or ModifierExtractor()
        self.llm = llm
        self.ambiguity_threshold = ambiguity_threshold
        self.enable_llm_post_coordination = enable_llm_post_coordination # [修改] 保存参数

    @staticmethod
    def _ph_mid(p):
        if "char_start" in p and "char_end" in p:
            return int((p["char_start"] + p["char_end"]) / 2)
        return 0

    @staticmethod
    def _mod_mid(m):
        return int((m.char_start + m.char_end) / 2)

    def _rule_bind(self, phenotypes, modifiers):
        bindings = {i: {} for i in range(len(phenotypes))}
        ambiguous = False

        if not phenotypes or not modifiers:
            return bindings, False

        mods_by_slot = {}
        for m in modifiers:
            mods_by_slot.setdefault(m.slot, []).append(m)

        for i, ph in enumerate(phenotypes):
            ph_mid = self._ph_mid(ph)
            for slot, mods in mods_by_slot.items():

                dist_list = []
                for m in mods:
                    d = abs(self._mod_mid(m) - ph_mid)
                    dist_list.append((d, m))
                dist_list.sort(key=lambda x: x[0])

                if not dist_list:
                    continue

                top1_dist, top1_mod = dist_list[0]
                if len(dist_list) > 1:
                    top2_dist, _ = dist_list[1]
                    if abs(top1_dist - top2_dist) <= self.ambiguity_threshold:
                        ambiguous = True

                bindings[i][slot] = top1_mod.value

        return bindings, ambiguous

    def annotate_sentence(self, sentence: str, phenotypes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not phenotypes:
            return phenotypes

        modifiers = self.extractor.extract(sentence)
        if not modifiers:
            return phenotypes

        # 单表型：直接赋值
        if len(phenotypes) == 1:
            ph = phenotypes[0]
            for m in modifiers:
                ph[m.slot] = m.value
            return phenotypes

        # 多表型：rule-based first
        rule_bindings, ambiguous = self._rule_bind(phenotypes, modifiers)

        # 触发 LLM 重决策
        # [修改] 增加开关检查：
        # 只有在 ambiguous=True 且 llm存在 且 开关enable_llm_post_coordination=True 时才调用
        if ambiguous and self.llm is not None and self.enable_llm_post_coordination:
            llm_bind = self.llm.bind(sentence, phenotypes, modifiers)
            for idx, slotmap in llm_bind.items():
                rule_bindings[idx].update(slotmap)

        # 写回 phenotype
        for i, ph in enumerate(phenotypes):
            for slot, value in rule_bindings[i].items():
                if slot in VALUE_DOMAINS and value in VALUE_DOMAINS[slot]:
                    ph[slot] = value

        return phenotypes