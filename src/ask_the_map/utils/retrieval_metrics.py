#!/usr/bin/env python3
"""
retrieval_metrics.py

Utilities for evaluating ranked retrieval results.

Metrics:
- Precision@K
- Recall@K
- Hits@K
- Average Precision (AP)
- DCG / NDCG
- Rank* / R~ normalized average rank

Important:
Ask-the-Map may rank image-level rows, while validation is usually done at
source_id / record level. Therefore duplicate IDs are removed while preserving
their first occurrence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


class RetrievalMetrics:
    def __init__(self, ranked_ids: Sequence[Any], relevant_ids: Iterable[Any]):
        self.ranked_ids = self._normalize_id_list(ranked_ids)
        self.relevant_ids = self._normalize_id_set(relevant_ids)

        self.N = len(self.ranked_ids)
        self.N_rel = len(self.relevant_ids)

        self.rank_lookup = self._build_rank_lookup()
        self.found_ranks = self._get_relevant_ranks()

    # -------------------------------------------------
    # ID normalization
    # -------------------------------------------------
    @staticmethod
    def normalize_id(x: Any) -> Optional[str]:
        if x is None:
            return None

        s = str(x).strip()

        if s == "":
            return None

        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except ValueError:
            pass

        return s

    @classmethod
    def _normalize_id_list(cls, ids: Iterable[Any]) -> List[str]:
        """
        Normalize IDs and remove duplicates while preserving first occurrence.

        This matters because Ask-the-Map often ranks image-level rows, but weak
        validation is usually performed at CommuniMap source_id / record level.
        """
        out: List[str] = []
        seen: Set[str] = set()

        for x in ids:
            nx = cls.normalize_id(x)

            if nx is None:
                continue

            if nx in seen:
                continue

            seen.add(nx)
            out.append(nx)

        return out

    @classmethod
    def _normalize_id_set(cls, ids: Iterable[Any]) -> Set[str]:
        out: Set[str] = set()

        for x in ids:
            nx = cls.normalize_id(x)

            if nx is not None:
                out.add(nx)

        return out

    @classmethod
    def extract_ranked_ids(
        cls,
        results: Sequence[dict],
        id_field: str = "source_id",
    ) -> List[str]:
        """
        Extract normalized ranked IDs from search results.

        Duplicates are removed while preserving the first occurrence.
        """
        ranked_ids: List[str] = []
        seen: Set[str] = set()

        for r in results:
            rid = cls.normalize_id(r.get(id_field))

            if rid is None:
                continue

            if rid in seen:
                continue

            seen.add(rid)
            ranked_ids.append(rid)

        return ranked_ids

    # -------------------------------------------------
    # Rank helpers
    # -------------------------------------------------
    def _build_rank_lookup(self) -> Dict[str, int]:
        """
        Build {item_id: 1-based rank}.
        """
        return {
            item_id: rank
            for rank, item_id in enumerate(self.ranked_ids, start=1)
        }

    def _get_relevant_ranks(self) -> List[int]:
        found = [
            self.rank_lookup[item_id]
            for item_id in self.relevant_ids
            if item_id in self.rank_lookup
        ]
        return sorted(found)

    # -------------------------------------------------
    # Basic metrics
    # -------------------------------------------------
    def hits_at_k(self, k: int) -> int:
        if k <= 0:
            return 0

        topk = self.ranked_ids[:k]
        return sum(1 for item_id in topk if item_id in self.relevant_ids)

    def precision_at_k(self, k: int) -> float:
        if k <= 0:
            return 0.0

        return self.hits_at_k(k) / k

    def recall_at_k(self, k: int) -> float:
        if self.N_rel == 0:
            return 0.0

        return self.hits_at_k(k) / self.N_rel

    # -------------------------------------------------
    # Average Precision
    # -------------------------------------------------
    def average_precision(self) -> float:
        if self.N_rel == 0:
            return 0.0

        hits = 0
        precisions: List[float] = []

        for rank, item_id in enumerate(self.ranked_ids, start=1):
            if item_id in self.relevant_ids:
                hits += 1
                precisions.append(hits / rank)

        return sum(precisions) / self.N_rel if precisions else 0.0

    # -------------------------------------------------
    # DCG / NDCG
    # -------------------------------------------------
    def dcg_at_k(self, k: Optional[int] = None) -> float:
        if k is None:
            k = self.N

        k = max(0, min(k, self.N))
        dcg = 0.0

        for rank, item_id in enumerate(self.ranked_ids[:k], start=1):
            if item_id in self.relevant_ids:
                dcg += 1.0 / math.log2(rank + 1)

        return dcg

    def idcg_at_k(self, k: Optional[int] = None) -> float:
        if self.N_rel == 0:
            return 0.0

        if k is None:
            k = self.N

        ideal_hits = min(self.N_rel, k)

        return sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

    def ndcg_at_k(self, k: Optional[int] = None) -> float:
        dcg = self.dcg_at_k(k)
        idcg = self.idcg_at_k(k)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    # -------------------------------------------------
    # Rank* / R~
    # -------------------------------------------------
    def rank_star(self) -> Optional[float]:
        """
        R~ / Rank* normalized average-rank metric.

        R~ = (sum(relevant ranks) - ideal rank sum) / (N * N_rel)

        Interpretation:
        - 0.0 = perfect ranking
        - approximately 0.5 = random ranking
        - closer to 1.0 = worse ranking
        """
        if self.N <= 0 or self.N_rel <= 0:
            return None

        if len(self.found_ranks) != self.N_rel:
            return None

        ideal_sum = (self.N_rel * (self.N_rel + 1)) / 2.0

        return (sum(self.found_ranks) - ideal_sum) / (self.N * self.N_rel)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    def summary(self, k_list: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if k_list is None:
            k_list = [5, 10, 20, 50, 100, 200, 500]

        valid_k_list = sorted(
            set(
                max(1, min(int(k), max(1, self.N)))
                for k in k_list
            )
        )

        precision = {k: self.precision_at_k(k) for k in valid_k_list}
        recall = {k: self.recall_at_k(k) for k in valid_k_list}
        hits = {k: self.hits_at_k(k) for k in valid_k_list}
        ndcg_k = {k: self.ndcg_at_k(k) for k in valid_k_list}

        return {
            "N": self.N,
            "N_rel": self.N_rel,
            "found_relevant": len(self.found_ranks),
            "found_ranks": self.found_ranks,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "hits_at_k": hits,
            "average_precision": self.average_precision(),
            "ndcg": self.ndcg_at_k(),
            "ndcg_at_k": ndcg_k,
            "rank_star": self.rank_star(),
            "rank_tilde": self.rank_star(),
            "mean_rank": (
                sum(self.found_ranks) / len(self.found_ranks)
                if self.found_ranks
                else None
            ),
            "median_rank": (
                self._median(self.found_ranks)
                if self.found_ranks
                else None
            ),
            "best_rank": min(self.found_ranks) if self.found_ranks else None,
            "worst_rank": max(self.found_ranks) if self.found_ranks else None,
        }

    def print_summary(self, k_list: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        s = self.summary(k_list)

        print("\n[METRICS] Ranking summary")
        print("-------------------------")
        print(f"N ranked items       : {s['N']}")
        print(f"N relevant items     : {s['N_rel']}")
        print(f"Relevant retrieved   : {s['found_relevant']} / {s['N_rel']}")

        print("\n[METRICS] Precision / Recall / NDCG by K")
        print("----------------------------------------")
        for k in sorted(s["precision_at_k"]):
            print(
                f"k={k:4d}  "
                f"Precision@{k}={s['precision_at_k'][k]:.6f}  "
                f"Recall@{k}={s['recall_at_k'][k]:.6f}  "
                f"NDCG@{k}={s['ndcg_at_k'][k]:.6f}  "
                f"Hits@{k}={s['hits_at_k'][k]}"
            )

        print("\n[METRICS] Global metrics")
        print("------------------------")
        print(f"Average Precision AP : {s['average_precision']:.6f}")
        print(f"NDCG                 : {s['ndcg']:.6f}")

        print("\n[METRICS] Rank summary")
        print("----------------------")

        if s["mean_rank"] is not None:
            print(f"Mean relevant rank   : {s['mean_rank']:.3f}")
            print(f"Median relevant rank : {s['median_rank']:.3f}")
            print(f"Best relevant rank   : {s['best_rank']}")
            print(f"Worst relevant rank  : {s['worst_rank']}")
        else:
            print("No relevant items were retrieved.")

        if s["rank_star"] is not None:
            print(f"R~ / Rank*           : {s['rank_star']:.6f}")
            print("Interpretation       : 0 is perfect, ~0.5 is random, closer to 1 is worse")
        else:
            print("R~ / Rank*           : not computed exactly")
            print("Reason               : not all relevant items were present in the ranking")

        print()

        return s

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    @staticmethod
    def _median(xs: Sequence[int]) -> Optional[float]:
        if not xs:
            return None

        ys = sorted(xs)
        n = len(ys)
        mid = n // 2

        if n % 2 == 1:
            return float(ys[mid])

        return 0.5 * (ys[mid - 1] + ys[mid])


if __name__ == "__main__":
    ranked_ids_demo = ["a", "b", "a", "c", "d", "e", "f", "g"]
    relevant_ids_demo = {"a", "c", "f"}

    metrics = RetrievalMetrics(
        ranked_ids=ranked_ids_demo,
        relevant_ids=relevant_ids_demo,
    )

    metrics.print_summary(k_list=[1, 3, 5, 7])