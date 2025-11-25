from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import Counter


# 断言优先级：absent > uncertain > historical > present
ASSERTION_PRIORITY: Dict[str, int] = {
    "absent": 0,
    "uncertain": 1,
    "historical": 2,
    "family": 3,     # family 一般单独使用，不直接映射 excluded
    "present": 4,
}

# 修饰槽位
MODIFIER_SLOTS: List[str] = [
    "onset",
    "severity",
    "laterality",
    "frequency",
    "progression",
]

SEVERITY_ORDER: Dict[str, int] = {"mild": 0, "moderate": 1, "severe": 2}
FREQUENCY_ORDER: Dict[str, int] = {
    "occasional": 0,
    "intermittent": 1,
    "frequent": 2,
    "recurrent": 3,
}
ONSET_ORDER: Dict[str, int] = {
    "neonatal": 0,
    "infancy": 1,
    "childhood": 2,
    "adult": 3,
    "congenital": 0,
}
PROGRESSION_ORDER: Dict[str, int] = {
    "stable": 0,
    "improving": 1,
    "worsening": 2,
    "progressive": 2,
    "deteriorating": 2,
}


@dataclass
class AggregationStats:
    total_spans: int = 0
    total_phenotypes: int = 0
    by_assertion: Dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in ASSERTION_PRIORITY.keys()}
    )


class PhenotypeAggregator:
    def __init__(self):
        self.stats = AggregationStats()

    def _merge_assertions(self, assertions: List[str]) -> str:
        valid = [a for a in assertions if a in ASSERTION_PRIORITY]
        if not valid:
            return "present"
        return min(valid, key=lambda a: ASSERTION_PRIORITY[a])

    def _merge_modifier_slot(self, slot: str, values: List[str]) -> Optional[str]:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]

        cnt = Counter(vals)
        most_common_val, freq = cnt.most_common(1)[0]
        if freq > 1:
            return most_common_val

        order_map: Dict[str, int] = {}
        if slot == "severity":
            order_map = SEVERITY_ORDER
        elif slot == "frequency":
            order_map = FREQUENCY_ORDER
        elif slot == "onset":
            order_map = ONSET_ORDER
        elif slot == "progression":
            order_map = PROGRESSION_ORDER

        if order_map:
            filtered = [v for v in vals if v in order_map]
            if filtered:
                return max(filtered, key=lambda v: order_map[v])

        return vals[0]

    def aggregate_note(
        self,
        spans: List[Dict[str, Any]],
        note_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not spans:
            return {"note_id": note_id, "phenotypes": []}

        self.stats.total_spans += len(spans)

        if note_id is None and "note_id" in spans[0]:
            note_id = spans[0]["note_id"]

        # --- 1. 按 hp_id 分组（兼容 C 阶段的 id 字段） ---
        by_hp: Dict[str, List[Dict[str, Any]]] = {}
        for sp in spans:
            hp_id = sp.get("hp_id") or sp.get("id")   # 🔴 关键修改在这里
            if not hp_id:
                continue
            by_hp.setdefault(hp_id, []).append(sp)

        phenotypes_ir: List[Dict[str, Any]] = []

        # --- 2. 对每个 hp_id 进行聚合 ---
        for hp_id, group in by_hp.items():
            labels = [g.get("label") for g in group if g.get("label")]
            label = labels[0] if labels else hp_id

            assertions = [g.get("assertion", "present") for g in group]
            merged_assertion = self._merge_assertions(assertions)
            self.stats.by_assertion[merged_assertion] += 1

            excluded = merged_assertion == "absent"

            merged_mods: Dict[str, Optional[str]] = {}
            for slot in MODIFIER_SLOTS:
                vals = [g.get(slot) for g in group if g.get(slot) is not None]
                merged_mods[slot] = self._merge_modifier_slot(slot, vals)

            evidence_list: List[Dict[str, Any]] = []
            for g in group:
                ev = {
                    "sent_id": g.get("sent_id"),
                    "span_text": g.get("span_text") or g.get("mention"),
                    "char_start": g.get("char_start"),
                    "char_end": g.get("char_end"),
                }
                evidence_list.append(ev)

            phenotype_ir = {
                "hp_id": hp_id,
                "label": label,
                "assertion": merged_assertion,
                "excluded": excluded,
                "onset": merged_mods["onset"],
                "severity": merged_mods["severity"],
                "laterality": merged_mods["laterality"],
                "frequency": merged_mods["frequency"],
                "progression": merged_mods["progression"],
                "evidence": evidence_list,
            }
            phenotypes_ir.append(phenotype_ir)

        self.stats.total_phenotypes += len(phenotypes_ir)

        return {
            "note_id": note_id,
            "phenotypes": phenotypes_ir,
        }