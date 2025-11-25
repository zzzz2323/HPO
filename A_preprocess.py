import re
from typing import List, Dict, Any
import spacy
from transformers import AutoTokenizer, AutoModel
from global_llm import GlobalLLM

# 使用最轻量的分句器，不需要下载模型
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


class PreprocessorSpaCy:
    def __init__(self, abbrev_path: str, use_llm: bool):
        self.abbr_dict = self._load_abbrev(abbrev_path)
        self.use_llm = use_llm
        self.llm_trigger_count = 0

        if use_llm:
            pack = GlobalLLM.get()
            self.model = pack["model"]
            self.tokenizer = pack["tokenizer"]

    def _load_abbrev(self, path: str) -> Dict[str, str]:
        abbr = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                short, full = line.strip().split("\t")
                abbr[short.lower()] = full
        return abbr

    def _clean(self, text: str) -> str:
        text = text.replace("\u200b", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_abbrev_candidate(self, token: str) -> bool:
        """
        判断是否是可能需要展开的缩写
        """
        token = re.sub(r"[^\w]", "", token)
        if len(token) <= 2:
            return True
        if token.isupper() and len(token) <= 4:
            return True
        if token.lower() in self.abbr_dict:
            return True
        return False

    def _expand_with_llm(self, word: str, sent_text: str) -> str:
        if not self.use_llm:
            return word
        self.llm_trigger_count += 1
        token = re.sub(r"[^\w]", "", word)

        # 必须过滤非缩写的情况
        if not self._is_abbrev_candidate(token):
            return word

        prompt = (
            "You are a clinical abbreviation expander.\n"
            "Your job is to expand a medical abbreviation in the given sentence.\n\n"
            "Requirements:\n"
            "1. Output ONLY the expanded full form of the abbreviation.\n"
            "2. Do NOT output the abbreviation itself.\n"
            "3. Do NOT output any explanation or extra words.\n"
            "4. Do NOT use quotation marks.\n"
            "5. Output must be one short noun phrase only (no sentence, no period).\n\n"
            f"Sentence: {sent_text}\n"
            f"Abbreviation: {word}\n\n"
            "Now output only the expanded form:"
        )


        try:
            response, _ = self.model.chat(self.tokenizer, prompt)
            text = response.strip().strip('"').strip("'")
            if len(text.split()) > 8:
                return word
            return text
        except Exception:
            return word

    def _expand_sentence(self, sent_doc) -> str:
        out_tokens = []
        for token in sent_doc:
            core = re.sub(r"[^\w]", "", token.text).lower()
            if core in self.abbr_dict:
                out_tokens.append(self.abbr_dict[core])
            else:
                out_tokens.append(self._expand_with_llm(token.text, sent_doc.text))
        return " ".join(out_tokens)

    def run(self, note_text: str, note_id: str) -> List[Dict[str, Any]]:
        text = self._clean(note_text)
        doc = nlp(text)
        self.llm_trigger_count = 0
        results = []
        for i, sent in enumerate(doc.sents):
            raw = sent.text
            clean = self._clean(raw)
            expanded = self._expand_sentence(sent)

            results.append({
                "note_id": note_id,
                "sent_id": i,
                "raw": raw,
                "clean": clean,
                "expanded": expanded,
                "char_start": sent.start_char,
                "char_end": sent.end_char,
            })
        return results
