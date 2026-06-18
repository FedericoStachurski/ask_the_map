from pathlib import Path
import bisect
import json

import faiss
import numpy as np
import pandas as pd
import torch

from ask_the_map.scripts.make_map import build_map
from ask_the_map.utils.load_model import load_model


# =========================================================
# HELPERS
# =========================================================

def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no GPU is available.")

    return device


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def minmax_dict(score_dict):
    if not score_dict:
        return {}

    values = list(score_dict.values())
    lo, hi = min(values), max(values)

    if hi - lo < 1e-8:
        return {k: 0.0 for k in score_dict}

    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def reciprocal_rank(rank, rrf_k):
    return 1.0 / (rrf_k + rank)


def has_valid_image(item):
    img = item.get("primary_image") or item.get("image")
    return isinstance(img, str) and img.strip() != ""


def get_percentile(score, sorted_scores):
    if not sorted_scores:
        return 0.0

    pos = bisect.bisect_left(sorted_scores, score)
    return pos / len(sorted_scores)


# =========================================================
# PIPELINE
# =========================================================

class AskMapPipeline:
    def __init__(
        self,
        embedding_folder,
        device="auto",
        collapse_source_ids=False,
    ):
        self.embedding_folder = Path(embedding_folder).expanduser().resolve()
        self.prefix_name = self.embedding_folder.name
        self.device = resolve_device(device)
        self.collapse_source_ids = collapse_source_ids

        self.text_path = self.embedding_folder / f"{self.prefix_name}_text.npy"
        self.image_path = self.embedding_folder / f"{self.prefix_name}_image.npy"
        self.meta_path = self.embedding_folder / f"{self.prefix_name}_meta.json"

        if not self.text_path.exists():
            raise FileNotFoundError(f"Text embeddings not found: {self.text_path}")

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image embeddings not found: {self.image_path}")

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")

        print("\n[ASK-MAP PIPELINE]")
        print(f"Embedding folder: {self.embedding_folder}")
        print(f"Text embeddings:  {self.text_path}")
        print(f"Image embeddings: {self.image_path}")
        print(f"Metadata:         {self.meta_path}")
        print(f"Device:           {self.device}")

        self.text_embs = np.load(self.text_path).astype("float32")
        self.img_embs = np.load(self.image_path).astype("float32")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if len(self.meta) != self.text_embs.shape[0]:
            raise ValueError("Metadata length does not match text embeddings.")

        if len(self.meta) != self.img_embs.shape[0]:
            raise ValueError("Metadata length does not match image embeddings.")

        if self.text_embs.shape[1] != self.img_embs.shape[1]:
            raise ValueError("Text and image embeddings have different dimensions.")

        self.model_name = self.meta[0].get("model_name")

        if not self.model_name:
            raise ValueError(
                "No model_name found in metadata. "
                "Rebuild embeddings with the updated builder."
            )

        print(f"[MODEL] Loading model: {self.model_name}")
        self.model = load_model(self.model_name, device=self.device)

        self.text_embs_norm = normalize_rows(self.text_embs)
        self.img_embs_norm = normalize_rows(self.img_embs)

        self.index_text = faiss.IndexFlatIP(self.text_embs_norm.shape[1])
        self.index_text.add(self.text_embs_norm)

        self.index_img = faiss.IndexFlatIP(self.img_embs_norm.shape[1])
        self.index_img.add(self.img_embs_norm)

        print("[FAISS] Text index:", self.index_text.ntotal, "vectors")
        print("[FAISS] Image index:", self.index_img.ntotal, "vectors")
        print("[READY]\n")

    def embed_query(self, query: str) -> np.ndarray:
        truncated_q = " ".join(str(query).split()[:50])

        return self.model.encode(
            [truncated_q],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def search(
        self,
        query: str,
        k: int = 50,
        threshold: float = 0.0,
        w_text: float = 0.3,
        w_img: float = 0.7,
        fusion_type: str = "weighted",
        rrf_k: int = 60,
    ):
        qv = self.embed_query(query)

        if qv.shape[1] != self.index_text.d:
            raise ValueError(
                f"Query dim mismatch: query has {qv.shape[1]}, "
                f"index expects {self.index_text.d}."
            )

        k_search = max(1, min(int(k), len(self.meta)))

        d_text, i_text = self.index_text.search(qv, k_search)
        d_img, i_img = self.index_img.search(qv, k_search)

        scores_t = d_text[0]
        idxs_t = i_text[0]

        scores_i = d_img[0]
        idxs_i = i_img[0]

        score_text_dict = {
            int(idx): float(score)
            for idx, score in zip(idxs_t, scores_t)
            if idx >= 0
        }

        score_img_dict = {
            int(idx): float(score)
            for idx, score in zip(idxs_i, scores_i)
            if idx >= 0
        }

        rank_text_dict = {
            int(idx): rank
            for rank, idx in enumerate(idxs_t, start=1)
            if idx >= 0
        }

        rank_img_dict = {
            int(idx): rank
            for rank, idx in enumerate(idxs_i, start=1)
            if idx >= 0
        }

        candidate_idxs = set(score_text_dict).union(score_img_dict)

        norm_text_dict = minmax_dict(score_text_dict)
        norm_img_dict = minmax_dict(score_img_dict)

        text_norm_scores = sorted(norm_text_dict.values())
        img_norm_scores = sorted(norm_img_dict.values())

        fused = []
        skipped_no_image = 0

        for idx in candidate_idxs:
            item = self.meta[idx]

            if not has_valid_image(item):
                skipped_no_image += 1
                continue

            st = norm_text_dict.get(idx, 0.0)
            si = norm_img_dict.get(idx, 0.0)

            rt = rank_text_dict.get(idx)
            ri = rank_img_dict.get(idx)

            if fusion_type == "weighted":
                score = w_text * st + w_img * si

            elif fusion_type == "text":
                score = st

            elif fusion_type == "image":
                score = si

            elif fusion_type == "rrf":
                score = 0.0

                if rt is not None:
                    score += reciprocal_rank(rt, rrf_k)

                if ri is not None:
                    score += reciprocal_rank(ri, rrf_k)

            else:
                raise ValueError(f"Unsupported fusion type: {fusion_type}")

            if score < threshold:
                continue

            p_text = 100.0 * get_percentile(st, text_norm_scores)
            p_img = 100.0 * get_percentile(si, img_norm_scores)

            fused.append((idx, score, st, si, p_text, p_img))

        print(f"[SEARCH] Skipped {skipped_no_image} candidates with no image.")

        fused.sort(key=lambda x: x[1], reverse=True)

        results = []

        for rank, (idx, score, st, si, p_text, p_img) in enumerate(fused, start=1):
            item = self.meta[idx]

            results.append(
                {
                    "rank": rank,
                    "idx": int(idx),
                    "score": float(score),
                    "score_text": float(st),
                    "score_img": float(si),
                    "p_text": float(p_text),
                    "p_img": float(p_img),
                    "id": item.get("id"),
                    "source_id": str(item.get("source_id", item.get("id"))),
                    "image_index": item.get("image_index"),
                    "media_column": item.get("media_column"),
                    "text": item.get("text", ""),
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "image": item.get("primary_image"),
                    "primary_image": item.get("primary_image"),
                    "CREATED_AT": item.get("CREATED_AT"),
                    "submission_date": item.get("submission_date"),
                }
            )

        if self.collapse_source_ids:
            seen = set()
            unique_results = []

            for r in results:
                sid = r["source_id"]

                if sid in seen:
                    continue

                seen.add(sid)
                unique_results.append(r)

            results = unique_results

            for new_rank, r in enumerate(results, start=1):
                r["rank"] = new_rank

        return results[:k_search]

    def run(
        self,
        query: str,
        k: int = 50,
        threshold: float = 0.0,
        w_text: float = 0.3,
        w_img: float = 0.7,
        fusion_type: str = "weighted",
        rrf_k: int = 60,
    ):
        results = self.search(
            query=query,
            k=k,
            threshold=threshold,
            w_text=w_text,
            w_img=w_img,
            fusion_type=fusion_type,
            rrf_k=rrf_k,
        )

        results_df = pd.DataFrame(results)

        if len(results) == 0:
            return results_df, None, results

        fmap = build_map(results)

        return results_df, fmap, results