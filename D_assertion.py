from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal, Tuple
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import json

from global_llm import GlobalLLM

# 全部可能的断言标签
ASSERTION_LABELS = ["present", "absent", "uncertain", "family", "historical"]
AssertionLabel = Literal["present", "absent", "uncertain", "family", "historical"]


# ============================================================
# 1. NegEx / ConText
# ============================================================
class RuleEngine:
    """
    ConText 风格规则
    """
    def __init__(self, neg_window: int = 8):
        self.neg_window = neg_window

        self.family_cues = [
            "family history", "fhx", "mother has", "father has",
            "sister has", "brother has", "parents have"
        ]
        self.hist_cues = [
            "history of", "hx of", "previous", "prior",
            "remote", "in childhood", "as a child"
        ]
        self.uncertain_cues = [
            "possible", "possibly", "suspected", "suspicion of",
            "likely", "cannot exclude", "rule out", "r/o", "might be"
        ]
        self.neg_tokens = ["no", "denies", "without", "never", "none"]
        self.neg_phrases = ["no evidence of", "negative for", "free of"]

    def apply(self, mention: str, context: str) -> Optional[AssertionLabel]:
        text = context.lower()
        m = mention.lower()

        # family
        for cue in self.family_cues:
            if cue in text:
                return "family"

        # historical
        for cue in self.hist_cues:
            if cue in text:
                return "historical"

        tokens = text.split()
        m_idx = None
        for i, tok in enumerate(tokens):
            if m.split()[0] in tok:
                m_idx = i
                break

        # negation (token-level)
        if m_idx is not None:
            for i, tok in enumerate(tokens):
                if tok in self.neg_tokens:
                    if 0 <= (m_idx - i) <= self.neg_window:
                        return "absent"

        # phrase negation
        for phrase in self.neg_phrases:
            if phrase in text and m in text:
                if text.index(phrase) < text.index(m):
                    return "absent"

        # uncertainty
        for cue in self.uncertain_cues:
            if cue in text:
                return "uncertain"

        return None


# ============================================================
# 2. 模型 (BERT)
# ============================================================
class AssertionSentenceClassifier:
    """
    轻量 BERT 句子分类器
    """
    def __init__(self,
                 model_name: str = "/public/home/202230275320/medicoding/models/distill_bert",
                 device: Optional[str] = None):

        self.label2id = {lab: i for i, lab in enumerate(ASSERTION_LABELS)}
        self.id2label = {i: lab for lab, i in self.label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(ASSERTION_LABELS)
        )

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def _build_input(self, mention: str, context: str):
        return f"[MENTION] {mention} [CONTEXT] {context}"

    def predict_proba(self, mention: str, context: str):
        text = self._build_input(mention, context)

        enc = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}

    def predict(self, mention: str, context: str) -> Tuple[str, float, Dict[str, float]]:
        probs = self.predict_proba(mention, context)
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        return label, probs[label], probs


# ============================================================
# 3. LLM
# ============================================================
class GlmEdgeAssertionLLM:
    """
    使用 GlobalLLM 单例共享模型
    """

    def __init__(self, model=None, tokenizer=None):
        if model is None or tokenizer is None:
            pack = GlobalLLM.get()
            self.model = pack["model"]
            self.tokenizer = pack["tokenizer"]
        else:
            self.model = model
            self.tokenizer = tokenizer

    def _extract_json(self, text: str):
        """
        更鲁棒的 JSON 提取方式（与 C 阶段一致）
        """
        t = text.strip()

        if t.startswith("```"):
            t = t.strip("`")
            idx = t.find("{")
            if idx != -1:
                t = t[idx:]

        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no json")

        return json.loads(t[start: end + 1])

    def _build_prompt(self, mention: str, context: str):
        return (
            "You are a clinical NLP assertion classifier.\n\n"
            "Your task:\n"
            "- Read the finding and its sentence context.\n"
            "- Select exactly one assertion label.\n"
            "- Allowed labels: present, absent, uncertain, family, historical.\n\n"
            "Output rules (IMPORTANT):\n"
            "- Output MUST be exactly one valid JSON object.\n"
            "- JSON schema: {\"label\":\"...\",\"reason\":\"...\",\"confidence\":0.0}\n"
            "- Do NOT output explanation outside JSON.\n"
            "- Do NOT add any extra text.\n"
            "- Do NOT repeat the instructions.\n"
            "- Do NOT output multiple JSON objects.\n\n"
            "JSON output template (copy structure exactly):\n"
            "{\n"
            "  \"label\": \"present\",\n"
            "  \"reason\": \"classification\",\n"
            "  \"confidence\": 0.90\n"
            "}\n\n"
            "Fill the template with the correct label.\n\n"
            f"Finding: {mention}\n"
            f"Context: {context}\n\n"
            "Now output ONLY the JSON object:"
        )


    def choose_assertion(self, mention: str, context: str):
        prompt = self._build_prompt(mention, context)

        try:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[])
            parsed = self._extract_json(response)
        except Exception as e:
            return {"label": "present", "reason": f"llm_error:{e}", "confidence": 0.55}

        label = parsed.get("label", "present")
        if label not in ASSERTION_LABELS:
            label = "present"

        return {
            "label": label,
            "reason": parsed.get("reason", "glm_choice"),
            "confidence": float(parsed.get("confidence", 0.55))
        }


# ============================================================
# 4. Orchestrator
# ============================================================
@dataclass
class AssertionStats:
    total: int = 0
    by_method: Dict[str, int] = field(default_factory=lambda: {"rule": 0, "model": 0, "llm": 0})
    by_label: Dict[str, int] = field(default_factory=lambda: {lab: 0 for lab in ASSERTION_LABELS})
    confusion: Dict[str, Dict[str, int]] = field(default_factory=lambda:
        {pred: {gold: 0 for gold in ASSERTION_LABELS} for pred in ASSERTION_LABELS}
    )


class AssertionClassifier:
    """
    [D] 断言分类：
        rule → ml_model → llm
    """

    def __init__(self,
                 rule_engine: Optional[RuleEngine] = None,
                 ml_model: Optional[AssertionSentenceClassifier] = None,
                 llm: Optional[GlmEdgeAssertionLLM] = None,
                 model_conf_threshold: float = 0.65,
                 enable_llm_assertion: bool = True): # [修改] 增加控制参数

        self.rule_engine = rule_engine or RuleEngine()
        self.ml_model = ml_model   # ⚠️ 推荐在 main 里传 None
        self.llm = llm

        self.model_conf_threshold = model_conf_threshold
        self.enable_llm_assertion = enable_llm_assertion # [修改] 保存参数
        self.stats = AssertionStats()

    @staticmethod
    def _build_context(sent: str, prev_sent: Optional[str], next_sent: Optional[str]):
        parts = []
        if prev_sent:
            parts.append(prev_sent.strip())
        parts.append(sent.strip())
        if next_sent:
            parts.append(next_sent.strip())
        return " ".join(parts)

    def _llm_decide(self, mention: str, context: str):
        if self.llm is None:
            return {"label": "present", "reason": "no_llm", "confidence": 0.5}
        return self.llm.choose_assertion(mention, context)

    def classify_span(self,
                      mention: str,
                      sent: str,
                      prev_sent: Optional[str] = None,
                      next_sent: Optional[str] = None,
                      gold_label: Optional[str] = None) -> Dict[str, Any]:

        self.stats.total += 1
        context = self._build_context(sent, prev_sent, next_sent)

        # ---- 1. rule ----
        rule_label = self.rule_engine.apply(mention, context)
        if rule_label is not None:
            label = rule_label
            method = "rule"
            conf = 0.9
            reason = "rule_engine"

        else:
            # ---- 2. model ----
            if self.ml_model is not None:
                ml_label, ml_conf, _ = self.ml_model.predict(mention, context)
            else:
                ml_label, ml_conf = "present", 0.0

            # [修改] 逻辑判断：
            # 如果模型置信度高 -> 使用模型
            # 或者 如果 LLM 被禁用 (enable_llm_assertion=False) -> 强制使用模型结果（即使置信度低）
            if ml_conf >= self.model_conf_threshold or not self.enable_llm_assertion:
                label = ml_label
                method = "model"
                conf = ml_conf
                reason = "model_confident" if ml_conf >= self.model_conf_threshold else "model_fallback_no_llm"
            else:
                # ---- 3. llm ----
                # 只有在置信度低 且 enable_llm_assertion=True 时才进入这里
                llm_out = self._llm_decide(mention, context)
                label = llm_out["label"]
                method = "llm"
                conf = llm_out["confidence"]
                reason = llm_out["reason"]

        if label not in ASSERTION_LABELS:
            label = "present"

        # stats
        self.stats.by_method[method] += 1
        self.stats.by_label[label] += 1

        if gold_label in ASSERTION_LABELS:
            self.stats.confusion[label][gold_label] += 1

        return {
            "assertion": label,
            "method": method,
            "confidence": conf,
            "reason": reason
        }