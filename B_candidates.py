import re
import os
from typing import List, Dict, Any
# from pronto import Ontology  <-- 删掉这一行，不再需要它了
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy

# ====================== 1. 辅助工具 (保持不变) ======================

STOP_WORDS = {
    # 基础功能词
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "and", 
    "or", "but", "no", "not", "is", "are", "was", "were", "be", "has", "have", 
    "had", "can", "will", "may", "should", "would", "either", "who", "which",
    
    # 临床背景词（从日志中发现）
    "patients", "patient", "present", "presents", "additional", "type", "series",
    "identified", "cases", "available", "suggest", "analysis", "strategy", "fulfil",
    "criter", "instance", "instances",
    
    # 通用名词和非特异性 HPO 概念
    "history", "report", "review", "findings", "physical", "examination", 
    "notes", "record", "documented", "child", "infant", "adult", "age", "year", 
    "old", "male", "female", "mother", "father", "family", "member", "onset", 
    "time", "mode", "inheritance", "syndrome", "disease", "disorder", "condition", 
    "sign", "symptom", "feature", "abnormality", "finding", "process", 
    "measurement", "clinical", "tumours", "tumour", "material"
}

# 2. 定义不应成为 Span 起点的词性 (POS)
# VERB (动词), AUX (助动词), ADP (介词), PRON (代词), CCONJ/SCONJ (连词), DET (限定词), ADV (副词)
ILLEGAL_POS_START = {"VERB", "AUX", "ADP", "PRON", "CCONJ", "SCONJ", "DET", "ADV"}

# 3. 定义 Span 内部不允许的标点
INVALID_CHARS = set("()[]{}<>,:;?./\\|!@#$%^&*")


def get_candidate_spans(text: str, nlp_model) -> List[str]:
    spans = []
    doc = nlp_model(text)
    
    # 提取 tokens (包含词性和文本)
    tokens = doc
    
    # --- A. 优先级最高：Spacy Noun Chunks (句法结构) ---
    # Noun Chunks 通常是高质量的候选
    if doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            
            # 过滤掉包含非法标点或纯数字/纯停用词的 Chunk
            if any(char in INVALID_CHARS for char in chunk_text):
                continue
            
            # 确保 Span 不以强停用词开头 (已修正)
            if chunk[0].lower_ in STOP_WORDS: 
                continue

            spans.append(chunk_text)

    # --- B. 优先级次之：N-gram 滑动窗口（补漏）---
    max_len = 6
        
    for i in range(len(tokens)):
        start_token = tokens[i]
        
        # 1. 严格过滤 Span 起点：与之前一样，过滤掉不应开始 Span 的词
        if start_token.pos_ in ILLEGAL_POS_START:
            continue
        if start_token.lower_ in STOP_WORDS:
            continue
        if any(char in INVALID_CHARS for char in start_token.text):
            continue

        # 循环遍历 Span 的结束位置 j
        for j in range(i, min(i + max_len, len(tokens))):
            end_token = tokens[j]
            
            # 遇到标点符号直接打断整个内循环，不再延伸
            if any(char in INVALID_CHARS for char in end_token.text):
                break
                
            # 获取当前 Span: tokens[i] 到 tokens[j]
            current_span_tokens = tokens[i:j+1]
            span_text = current_span_tokens.text.strip()
            
            # --- 过滤检查：如果 Span 不合法，则跳过 'append' ---
            
            # 2. 规则：Span 必须是名词或形容词（避免单独的动词或副词 Span）
            if len(current_span_tokens) == 1 and current_span_tokens[0].pos_ not in ['NOUN', 'ADJ']:
                continue

            # 3. 规则：Span 不能以强停用词或非法词性结尾
            if end_token.lower_ in STOP_WORDS or end_token.pos_ in ILLEGAL_POS_START:
                # 如果结尾是通用词，则跳过当前 Span 的添加，但继续尝试延伸到下一个词 (j+1)
                continue 

            # 4. 规则：最终 Span 长度检查
            if len(span_text) < 2:
                continue
                
            # Span 通过所有过滤，加入列表
            if span_text:
                spans.append(span_text)

    # --- C. 去重与清洗 ---
    unique_spans = []
    seen = set()
    for s in spans:
        clean = s.strip()
        key = clean.lower()
        
        # 最终过滤：避免长度过短或残留非法字符
        if len(key) < 3: continue
        if key in STOP_WORDS: continue
        if any(char in INVALID_CHARS for char in key): continue

        if key not in seen:
            seen.add(key)
            unique_spans.append(clean)
            
    return unique_spans


# ====================== 2. 修正后的 Lexicon (手动解析版) ======================

class HPOLexicon:
    def __init__(self, hp_obo_path: str, embed_model_path="/public/home/202230275320/medicoding/models/bge"):
        self.surface_forms = []
        self.hp_ids = []
        self.hp_labels = []
        
        # 黑名单与忽略ID
        self.blacklist = {
            "all", "other", "none", "finding", "abnormality", "disease", 
            "disorder", "syndrome", "phenotype", "mode of inheritance", 
            "clinical course", "past medical history"
        }
        self.ignore_ids = {"HP:0000001", "HP:0000118", "HP:0000005"}

        # === 手动加载 OBO ===
        print(f"Loading HPO Ontology (Manual Parse): {hp_obo_path} ...")
        self._load_obo_manual(hp_obo_path)
        print(f"HPO Lexicon ready: {len(self.surface_forms)} terms loaded.")

        # BM25
        print("Building BM25 index...")
        self.tokenized_corpus = [sf.split() for sf in self.surface_forms]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Embedding
        print("Loading BGE embedder...")
        self.embedder = SentenceTransformer(embed_model_path)
        self.vecs = self.embedder.encode(
            self.surface_forms,
            batch_size=512,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def _load_obo_manual(self, path):
        """
        手动解析 OBO 文件，跳过 pronto 库的 strict 检查
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        current_id = None
        current_name = None
        current_synonyms = []
        is_obsolete = False
        
        # 正则匹配 synonym 行: synonym: "TEXT" EXACT ...
        syn_pattern = re.compile(r'synonym: "(.*?)"')

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 遇到新 Term 或文件结束时，保存上一个
                if line == '[Term]':
                    self._add_entry(current_id, current_name, current_synonyms, is_obsolete)
                    # 重置状态
                    current_id = None
                    current_name = None
                    current_synonyms = []
                    is_obsolete = False
                    continue
                
                if line.startswith('id: HP:'):
                    current_id = line.split('id: ')[1].split()[0]
                elif line.startswith('name:'):
                    current_name = line.split('name: ')[1]
                elif line.startswith('synonym:'):
                    m = syn_pattern.search(line)
                    if m:
                        current_synonyms.append(m.group(1))
                elif line.startswith('is_obsolete: true'):
                    is_obsolete = True

            # 循环结束后保存最后一个
            self._add_entry(current_id, current_name, current_synonyms, is_obsolete)

    def _add_entry(self, hp_id, name, synonyms, is_obsolete):
        """
        辅助函数：应用过滤逻辑并存入列表
        """
        if not hp_id or not name:
            return
        if is_obsolete:
            return
        if hp_id in self.ignore_ids:
            return

        # 收集所有形式（name + synonyms）
        forms = set()
        forms.add(name)
        forms.update(synonyms)

        for text in forms:
            text_lower = text.lower().strip()
            
            # 过滤逻辑
            if len(text_lower) < 3: continue
            if text_lower in self.blacklist: continue
            
            # 存入
            self.surface_forms.append(text_lower)
            self.hp_ids.append(hp_id)
            self.hp_labels.append(name) # 统一存标准名，方便后续展示

    def hybrid_recall(self, mention: str, bm25_topk=50, final_topk=10):
        mention_lower = mention.lower().strip()

        # 1. BM25 初始召回
        scores = self.bm25.get_scores(mention_lower.split())
        bm25_idx = np.argsort(scores)[::-1][:bm25_topk]

        if len(bm25_idx) == 0:
            return [], []

        # 2. Embedding 相似度计算
        query_vec = self.embedder.encode(mention_lower, normalize_embeddings=True, convert_to_numpy=True)
        cand_vecs = self.vecs[bm25_idx]
        embed_scores = (cand_vecs @ query_vec.T).flatten()

        # 3. 排序
        ranked_local_idx = np.argsort(embed_scores)[::-1]
        ranked_global_idx = bm25_idx[ranked_local_idx]
        
        # 将 Embedding 分数复制到新的数组，用于修改
        ranked_scores = embed_scores[ranked_local_idx].copy() 

        # ====== 4. 关键修复：词汇完美匹配优先级覆盖 ======
        # 查找并标记所有与输入 Span 完美匹配的表面形式
        perfect_match_found = False
        
        for i, idx in enumerate(ranked_global_idx):
            # self.surface_forms 包含所有 HPO 概念的名称和同义词 (已 lower)
            if self.surface_forms[idx] == mention_lower:
                # 强制将完美匹配的项分数提升到 1.001
                ranked_scores[i] = 1.001 
                perfect_match_found = True
                
        # 如果找到了完美匹配，需要重新排序以确保 1.001 的分数排在最前面
        if perfect_match_found:
            new_order = np.argsort(ranked_scores)[::-1]
            ranked_global_idx = ranked_global_idx[new_order]
            ranked_scores = ranked_scores[new_order]

        # 5. 动态截断策略（现在只依赖分数，不再有硬性数量限制）
        valid_global_idx = []
        valid_scores = []
        
        for i, (idx, score) in enumerate(zip(ranked_global_idx, ranked_scores)):
            # ****** 关键修改点：移除 i >= final_topk 的硬性限制 ******
            
            # 截断条件：如果分数低于 0.65 且不是完美匹配 (1.001)，则停止
            if score < 0.65 and score < 1.001: 
                break 
            
            valid_global_idx.append(int(idx))
            valid_scores.append(float(score))

        return valid_global_idx, valid_scores

# ====================== 3. 主流程入口 (已设置 0.85 阈值) ======================

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
except:
    print("Warning: Spacy model not found, using blank model.")
    nlp = spacy.blank("en")

def generate_hpo_candidates(
    sentences: List[Dict[str, Any]],
    lexicon: HPOLexicon,
    bm25_topk: int = 100,
    final_topk: int = 20
):
    results = []
    for s in sentences:
        text = s["expanded"]
        sid = s["sent_id"]
        nid = s["note_id"]

        unique_spans = get_candidate_spans(text, nlp)

        for sp in unique_spans:
            # final_topk 的值现在对 hybrid_recall 内部不再起硬性限制作用
            idxs, scores = lexicon.hybrid_recall(sp, bm25_topk=bm25_topk, final_topk=final_topk) 
            
            if not idxs:
                continue
            
            # ****** 应用高置信度阈值 0.85 ******
            if scores[0] < 0.85: 
                continue

            cands = []
            for i, idx in enumerate(idxs):
                cands.append({
                    "id": lexicon.hp_ids[idx],
                    "label": lexicon.hp_labels[idx],
                    "surface": lexicon.surface_forms[idx],
                    "score": scores[i],
                    "match_type": "hybrid"
                })

            results.append({
                "note_id": nid,
                "sent_id": sid,
                "span_text": sp,
                "candidates": cands
            })

    return results