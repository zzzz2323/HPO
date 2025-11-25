"""
evaluate_gsc.py
---------------
评估 pipeline 输出的 IR (output.json)
对比 GSC / GSCplus 的 gold_spans 得到：
  - Mention-level F1 （span + concept）
  - Concept-level F1 （doc-level concept sets）
"""

import json
from typing import List, Dict, Any


# ============================================================
# 1) 读取 gold GSC 文件
# ============================================================

def load_gsc_gold(path: str):
    """
    返回结构：
    [
      {
        "doc_id": "100175",
        "text": "...",
        "gold_spans": [
            {"start": 10, "end": 20, "mention": "...", "hp_id": "HP:0001250"},
            ...
        ]
      },
      ...
    ]
    """
    docs = []
    current_id = None
    current_text = []
    gold_spans = []

    def flush():
        if current_id is None:
            return
        docs.append({
            "doc_id": current_id,
            "text": "\n".join(current_text).strip(),
            "gold_spans": gold_spans.copy()
        })

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if line.strip().isdigit():
                flush()
                current_id = line.strip()
                current_text = []
                gold_spans = []
                continue

            if not line.strip():
                continue

            parts = line.split()
            # gold span line
            if len(parts) >= 4 and parts[-1].startswith("HP:") and parts[0].isdigit():
                start = int(parts[0])
                end = int(parts[1])
                hp_id = parts[-1]
                mention = " ".join(parts[2:-1])
                gold_spans.append({
                    "start": start,
                    "end": end,
                    "hp_id": hp_id,
                    "mention": mention
                })
            else:
                # text line
                current_text.append(line)

    flush()
    return {d["doc_id"]: d for d in docs}


# ============================================================
# 2) Mention-level 对齐函数
# ============================================================

def match_spans(gold: Dict[str, Any], pred: Dict[str, Any]):
    """
    gold: {"start","end","hp_id"}
    pred: {"char_start","char_end","hp_id"}

    返回 True/False 是否严格匹配。
    """

    if pred["char_start"] is None or pred["char_end"] is None:
        return False

    return (
        gold["start"] == pred["char_start"] and
        gold["end"] == pred["char_end"] and
        gold["hp_id"] == pred["hp_id"]
    )


# ============================================================
# 3) F1 计算
# ============================================================

def compute_f1(tp: int, fp: int, fn: int):
    if tp == 0 and (fp == 0 and fn == 0):
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ============================================================
# 4) 计算 mention-level F1 和 concept-level F1
# ============================================================

def evaluate(gold_path: str, pred_path: str):

    # --------------- 加载 gold -----------------
    gold_docs = load_gsc_gold(gold_path)

    # --------------- 加载预测 IR（output.json） ---------------
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_docs = json.load(f)

    # 将 IR 转成 doc_id → phenotypes
    pred_docs_map = {d["note_id"]: d for d in pred_docs}

    # mention-level 计数器
    tp = fp = fn = 0

    # concept-level 计数器
    concept_tp = concept_fp = concept_fn = 0

    for doc_id, gdoc in gold_docs.items():
        g_spans = gdoc["gold_spans"]
        pred_doc = pred_docs_map.get(doc_id)

        if pred_doc is None:
            # 全部 FN
            fn += len(g_spans)
            concept_fn += len({s["hp_id"] for s in g_spans})
            continue

        p_phens = pred_doc["phenotypes"]

        # ========== mention-level 对齐 ==========
        # gold span list → predicted span list
        used_pred = set()

        for gs in g_spans:
            found = False
            for i, pp in enumerate(p_phens):
                ev_list = pp["evidence"]
                for ev in ev_list:
                    pred_span = {
                        "char_start": ev["char_start"],
                        "char_end": ev["char_end"],
                        "hp_id": pp["hp_id"]
                    }
                    if match_spans(gs, pred_span):
                        found = True
                        used_pred.add((i, ev["char_start"], ev["char_end"]))
                        break
                if found:
                    break

            if found:
                tp += 1
            else:
                fn += 1

        # FP = 在预测中出现，但未被 gold 匹配的 spans
        for pp in p_phens:
            for ev in pp["evidence"]:
                key = (pp["hp_id"], ev["char_start"], ev["char_end"])
                # 如果这个 span 没有出现在 gold 匹配集中，就是 FP
                matched = False
                for gs in g_spans:
                    span_g = {
                        "char_start": gs["start"],
                        "char_end": gs["end"],
                        "hp_id": gs["hp_id"]
                    }
                    if match_spans(gs, {
                        "char_start": ev["char_start"],
                        "char_end": ev["char_end"],
                        "hp_id": pp["hp_id"]
                    }):
                        matched = True
                        break
                if not matched:
                    fp += 1

        # ========== concept-level F1 ==========
        gold_hpos = {s["hp_id"] for s in g_spans}
        pred_hpos = {p["hp_id"] for p in p_phens}

        concept_tp += len(gold_hpos & pred_hpos)
        concept_fp += len(pred_hpos - gold_hpos)
        concept_fn += len(gold_hpos - pred_hpos)

    # mention-level F1
    mention_P, mention_R, mention_F1 = compute_f1(tp, fp, fn)

    # concept-level F1
    concept_P, concept_R, concept_F1 = compute_f1(concept_tp, concept_fp, concept_fn)

    result = {
        "mention_level": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": mention_P,
            "recall": mention_R,
            "f1": mention_F1
        },
        "concept_level": {
            "TP": concept_tp,
            "FP": concept_fp,
            "FN": concept_fn,
            "precision": concept_P,
            "recall": concept_R,
            "f1": concept_F1
        }
    }

    return result


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold", default="/root/HPO/GSCplus_test_gold.tsv", help="GSCplus_test_gold.tsv")
    parser.add_argument("--pred", default="/root/output3.json", help="pipeline output.json")

    args = parser.parse_args()

    res = evaluate(args.gold, args.pred)
    print(json.dumps(res, indent=2))
