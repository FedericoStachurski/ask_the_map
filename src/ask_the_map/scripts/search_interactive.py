#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ask_the_map.scripts.make_map import build_map
from ask_the_map.utils.load_model import load_model
from ask_the_map.utils.load_data_communimap import load_raw_dataframe
from ask_the_map.utils.retrieval_metrics import RetrievalMetrics


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no GPU is available.")

    return device


def safe_filename(s: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(s).strip()
    )
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80].strip("_") or "query"


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


def normalize_id(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def parse_k_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Ask-the-Map search over local embeddings."
    )

    parser.add_argument(
        "--embedding-folder",
        required=True,
        help="Folder containing <folder_name>_text.npy, <folder_name>_image.npy, <folder_name>_meta.json.",
    )

    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional original CommuniMap CSV/XLSX path for weak validation.",
    )

    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--w-text", type=float, default=0.7)
    parser.add_argument("--w-img", type=float, default=0.3)

    parser.add_argument(
        "--fusion",
        choices=["weighted", "rrf", "text", "image"],
        default="weighted",
    )

    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--validate-k-list",
        default="5,10,20,50,100,200,500",
    )

    parser.add_argument(
        "--collapse-source-ids",
        action="store_true",
        help="Keep only the best result per CommuniMap source_id.",
    )

    return parser.parse_args()


def build_relevance_set_from_table(
    meta,
    df,
    field_value_map,
    id_field="ID",
    label_name="validation",
    meta_id_field="source_id",
):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    id_field_norm = str(id_field).strip().lower()

    if id_field_norm not in df.columns:
        raise ValueError(
            f"ID field {id_field!r} not found. Available columns: {df.columns.tolist()}"
        )

    criteria = {
        str(field).strip().lower(): {str(v).strip().lower() for v in valid_values}
        for field, valid_values in field_value_map.items()
    }

    missing_fields = [field for field in criteria if field not in df.columns]
    if missing_fields:
        raise ValueError(
            f"Missing validation fields: {missing_fields}. "
            f"Available columns: {df.columns.tolist()}"
        )

    searchable_ids = set()

    for item in meta:
        if not has_valid_image(item):
            continue

        item_id = normalize_id(item.get(meta_id_field, item.get("id")))
        if item_id is not None:
            searchable_ids.add(item_id)

    is_relevant = pd.Series(False, index=df.index)

    for field, valid_values in criteria.items():
        col = df[field].astype(str).str.strip().str.lower()
        is_relevant = is_relevant | col.isin(valid_values)

    relevant_ids = set(
        df.loc[is_relevant, id_field_norm]
        .dropna()
        .map(normalize_id)
        .dropna()
    )

    overlap = relevant_ids.intersection(searchable_ids)

    print(f"[VALIDATE:{label_name}] Relevant IDs before searchable filter: {len(relevant_ids)}")
    print(f"[VALIDATE:{label_name}] Searchable IDs from meta: {len(searchable_ids)}")
    print(f"[VALIDATE:{label_name}] Final searchable relevant IDs: {len(overlap)}")

    return searchable_ids, overlap


def choose_validation_config():
    print("\n[VALIDATE] Choose validation target:")
    print("  1) trees")
    print("  2) flooding / puddles / standing water")

    choice = input("Validation target (1/2)> ").strip().lower()

    if choice in {"1", "tree", "trees"}:
        return {
            "label_name": "trees",
            "query": "trees",
            "field_value_map": {
                "TREE": {"yes"},
            },
        }

    if choice in {"2", "flood", "flooding", "puddle", "puddles", "water"}:
        return {
            "label_name": "flooding",
            "query": "puddles flooding standing water",
            "field_value_map": {
                "TYPE_OF_WATER_EVENT": {
                    "puddle",
                    "flooding",
                    "standing water",
                    "Puddle",
                    "Flooding",
                    "Standing Water",
                },
            },
        }

    print("[VALIDATE] Invalid choice. Defaulting to trees.")
    return {
        "label_name": "trees",
        "query": "trees",
        "field_value_map": {
            "TREE": {"yes"},
        },
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_validation_summary_csv(path: Path, summary, query, fusion_mode):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["metric", "value"])
        writer.writerow(["query", query])
        writer.writerow(["fusion_mode", fusion_mode])
        writer.writerow(["N", summary["N"]])
        writer.writerow(["N_rel", summary["N_rel"]])
        writer.writerow(["found_relevant", summary["found_relevant"]])
        writer.writerow(["average_precision", summary["average_precision"]])
        writer.writerow(["ndcg", summary["ndcg"]])
        writer.writerow(["R_tilde", summary.get("rank_tilde", summary.get("rank_star"))])
        writer.writerow(["rank_star", summary.get("rank_star")])
        writer.writerow(["mean_rank", summary["mean_rank"]])
        writer.writerow(["median_rank", summary["median_rank"]])
        writer.writerow(["best_rank", summary["best_rank"]])
        writer.writerow(["worst_rank", summary["worst_rank"]])

        for k in sorted(summary["precision_at_k"]):
            writer.writerow([f"precision_at_{k}", summary["precision_at_k"][k]])
            writer.writerow([f"recall_at_{k}", summary["recall_at_k"][k]])
            writer.writerow([f"hits_at_{k}", summary["hits_at_k"][k]])
            writer.writerow([f"ndcg_at_{k}", summary["ndcg_at_k"][k]])


def plot_validation_curves(path: Path, data_csv_path: Path, summary, query, fusion_mode):
    ks = sorted(summary["precision_at_k"].keys())

    rows = []
    for k in ks:
        rows.append(
            {
                "k": k,
                "precision_at_k": summary["precision_at_k"][k],
                "recall_at_k": summary["recall_at_k"][k],
                "ndcg_at_k": summary["ndcg_at_k"][k],
                "hits_at_k": summary["hits_at_k"][k],
            }
        )

    pd.DataFrame(rows).to_csv(data_csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, [r["precision_at_k"] for r in rows], marker="o", label="Precision@K")
    plt.plot(ks, [r["recall_at_k"] for r in rows], marker="o", label="Recall@K")
    plt.plot(ks, [r["ndcg_at_k"] for r in rows], marker="o", label="NDCG@K")
    plt.xlabel("K")
    plt.ylabel("Metric value")
    plt.title(f"Validation curves\nquery={query!r}, fusion={fusion_mode}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_rank_histogram(path: Path, data_csv_path: Path, ranks, query, fusion_mode):
    if not ranks:
        return

    pd.DataFrame({"rank": ranks}).to_csv(data_csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(ranks, bins=40, alpha=0.8)
    plt.xlabel("Rank of relevant items")
    plt.ylabel("Count")
    plt.title(f"Relevant rank distribution\nquery={query!r}, fusion={fusion_mode}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_cumulative_relevant_curve(
    path: Path,
    data_csv_path: Path,
    ranked_ids,
    relevant_ids,
    query,
    fusion_mode,
):
    ranked_ids_norm = [normalize_id(x) for x in ranked_ids if normalize_id(x) is not None]
    relevant_ids_norm = {normalize_id(x) for x in relevant_ids if normalize_id(x) is not None}

    if not ranked_ids_norm or not relevant_ids_norm:
        return

    rows = []
    hits = 0

    for rank, item_id in enumerate(ranked_ids_norm, start=1):
        if item_id in relevant_ids_norm:
            hits += 1

        rows.append(
            {
                "rank": rank,
                "source_id": item_id,
                "cumulative_relevant": hits,
            }
        )

    pd.DataFrame(rows).to_csv(data_csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot([r["rank"] for r in rows], [r["cumulative_relevant"] for r in rows])
    plt.xlabel("Rank position")
    plt.ylabel("Cumulative relevant retrieved")
    plt.title(f"Cumulative relevant-retrieved curve\nquery={query!r}, fusion={fusion_mode}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    args = parse_args()

    embedding_folder = Path(args.embedding_folder).expanduser().resolve()
    prefix_name = embedding_folder.name

    text_path = embedding_folder / f"{prefix_name}_text.npy"
    image_path = embedding_folder / f"{prefix_name}_image.npy"
    meta_path = embedding_folder / f"{prefix_name}_meta.json"

    if not text_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {text_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {image_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    validation_dir = embedding_folder / "validation"
    results_dir = embedding_folder / "results"
    maps_dir = embedding_folder / "maps"

    validation_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    validate_k_list = parse_k_list(args.validate_k_list)

    print("\n[SETUP]")
    print(f"Embedding folder: {embedding_folder}")
    print(f"Text embeddings:  {text_path}")
    print(f"Image embeddings: {image_path}")
    print(f"Metadata:         {meta_path}")
    print(f"Device:           {device}")
    print(f"Results folder:   {results_dir}")
    print(f"Maps folder:      {maps_dir}")
    print(f"Validation folder:{validation_dir}\n")

    text_embs = np.load(text_path).astype("float32")
    img_embs = np.load(image_path).astype("float32")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(meta) != text_embs.shape[0]:
        raise ValueError("Metadata length does not match text embeddings.")
    if len(meta) != img_embs.shape[0]:
        raise ValueError("Metadata length does not match image embeddings.")
    if text_embs.shape[1] != img_embs.shape[1]:
        raise ValueError("Text and image embeddings have different dimensions.")

    model_name = meta[0].get("model_name")
    if not model_name:
        raise ValueError("No model_name found in metadata. Rebuild embeddings with updated builder.")

    print(f"[MODEL] Loading model: {model_name}")
    model = load_model(model_name, device=device)

    text_embs_norm = normalize_rows(text_embs)
    img_embs_norm = normalize_rows(img_embs)

    index_text = faiss.IndexFlatIP(text_embs_norm.shape[1])
    index_text.add(text_embs_norm)

    index_img = faiss.IndexFlatIP(img_embs_norm.shape[1])
    index_img.add(img_embs_norm)

    print("[FAISS] Text index:", index_text.ntotal, "vectors, dim =", index_text.d)
    print("[FAISS] Image index:", index_img.ntotal, "vectors, dim =", index_img.d)

    df_validation = None
    if args.data_path:
        df_validation = load_raw_dataframe(args.data_path)
        print(f"[VALIDATE] Loaded raw validation table: {df_validation.shape}")

    def embed_query(q: str) -> np.ndarray:
        truncated_q = " ".join(q.split()[:50])
        return model.encode(
            [truncated_q],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def get_percentile(score, sorted_scores):
        if not sorted_scores:
            return 0.0
        pos = bisect.bisect_left(sorted_scores, score)
        return pos / len(sorted_scores)

    def search_multimodal(query, k, threshold, w_text, w_img, fusion_type, rrf_k):
        qv = embed_query(query)

        if qv.shape[1] != index_text.d:
            raise ValueError(
                f"Query dim mismatch: query has {qv.shape[1]}, index expects {index_text.d}."
            )

        k_search = max(1, min(int(k), len(meta)))

        d_text, i_text = index_text.search(qv, k_search)
        d_img, i_img = index_img.search(qv, k_search)

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
            item = meta[idx]

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

        for idx, score, st, si, p_text, p_img in fused:
            item = meta[idx]

            results.append(
                {
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
                }
            )

        if args.collapse_source_ids:
            seen = set()
            unique_results = []

            for r in results:
                sid = r["source_id"]
                if sid in seen:
                    continue

                seen.add(sid)
                unique_results.append(r)

            results = unique_results

        return results[:k_search]

    def run_validation(current_fusion):
        nonlocal df_validation

        if df_validation is None:
            print("[VALIDATE] No --data-path supplied, validation cannot run.")
            return

        cfg = choose_validation_config()
        label = cfg["label_name"]
        query = cfg["query"]

        searchable_ids, relevant_ids = build_relevance_set_from_table(
            meta=meta,
            df=df_validation,
            field_value_map=cfg["field_value_map"],
            id_field="ID",
            label_name=label,
            meta_id_field="source_id",
        )

        n = len(searchable_ids)
        n_rel = len(relevant_ids)

        if n == 0:
            print("[VALIDATE] No searchable IDs found.")
            return

        if n_rel == 0:
            print("[VALIDATE] No relevant IDs found.")
            return

        print(f"[VALIDATE] Running query: {query!r}")
        print(f"[VALIDATE] N={n}, N_rel={n_rel}")

        full_results = search_multimodal(
            query=query,
            k=len(meta),
            threshold=-1.0,
            w_text=args.w_text,
            w_img=args.w_img,
            fusion_type=current_fusion,
            rrf_k=args.rrf_k,
        )

        ranked_ids = [
            normalize_id(r.get("source_id"))
            for r in full_results
            if r.get("source_id") is not None
        ]

        metrics = RetrievalMetrics(
            ranked_ids=ranked_ids,
            relevant_ids=relevant_ids,
        )

        summary = metrics.print_summary(validate_k_list)

        base = f"{safe_filename(label)}_{safe_filename(query)}_{current_fusion}"

        summary_json = validation_dir / f"{base}_summary.json"
        summary_csv = validation_dir / f"{base}_summary.csv"
        ranked_csv = validation_dir / f"{base}_ranked_ids.csv"
        relevant_csv = validation_dir / f"{base}_relevant_ids.csv"

        save_json(
            summary_json,
            {
                "label": label,
                "query": query,
                "fusion": current_fusion,
                "k_list": validate_k_list,
                "R_tilde": summary.get("rank_tilde", summary.get("rank_star")),
                "rank_star": summary.get("rank_star"),
                "summary": summary,
            },
        )

        save_validation_summary_csv(summary_csv, summary, query, current_fusion)

        pd.DataFrame({"rank": range(1, len(ranked_ids) + 1), "source_id": ranked_ids}).to_csv(
            ranked_csv,
            index=False,
        )

        pd.DataFrame({"source_id": sorted(relevant_ids)}).to_csv(
            relevant_csv,
            index=False,
        )

        plot_validation_curves(
            validation_dir / f"{base}_curves.png",
            validation_dir / f"{base}_curves_data.csv",
            summary,
            query,
            current_fusion,
        )

        plot_rank_histogram(
            validation_dir / f"{base}_rank_hist.png",
            validation_dir / f"{base}_rank_hist_data.csv",
            metrics.found_ranks,
            query,
            current_fusion,
        )

        plot_cumulative_relevant_curve(
            validation_dir / f"{base}_cumulative.png",
            validation_dir / f"{base}_cumulative_data.csv",
            ranked_ids,
            relevant_ids,
            query,
            current_fusion,
        )

        print("\n[VALIDATE] Saved:")
        print(f"  {summary_json}")
        print(f"  {summary_csv}")
        print(f"  {ranked_csv}")
        print(f"  {relevant_csv}")
        print(f"  validation plots + plot data CSVs in {validation_dir}\n")

        print("[VALIDATE] Summary:")
        print(f"  AP:       {summary['average_precision']:.6f}")
        print(f"  NDCG:     {summary['ndcg']:.6f}")
        print(f"  R~:       {summary.get('rank_tilde', summary.get('rank_star'))}")
        print(f"  Found:    {summary['found_relevant']} / {summary['N_rel']}\n")

    print("\n[READY] Enter search queries.")
    print("Commands:")
    print("  k=<int>")
    print("  threshold=<float>")
    print("  fusion=weighted|rrf|text|image")
    print("  VALIDATE")
    print("  quit / exit")
    print()

    current_k = args.k
    current_threshold = args.threshold
    current_fusion = args.fusion

    while True:
        try:
            q = input(
                f"Query (k={current_k}, thr={current_threshold}, fusion={current_fusion})> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break

        if not q:
            continue

        if q.lower() in {"q", "quit", "exit"}:
            print("[EXIT]")
            break

        if q.upper() == "VALIDATE":
            run_validation(current_fusion)
            continue

        if q.startswith("k=") or q.startswith("k ="):
            try:
                current_k = int(q.split("=", 1)[1])
                print(f"[SET] k updated to {current_k}\n")
            except Exception:
                print("[ERROR] Invalid k.\n")
            continue

        if q.startswith("threshold=") or q.startswith("threshold ="):
            try:
                current_threshold = float(q.split("=", 1)[1])
                print(f"[SET] threshold updated to {current_threshold}\n")
            except Exception:
                print("[ERROR] Invalid threshold.\n")
            continue

        if q.startswith("fusion=") or q.startswith("fusion ="):
            try:
                new_fusion = q.split("=", 1)[1].strip().lower()
                if new_fusion not in {"weighted", "rrf", "text", "image"}:
                    raise ValueError
                current_fusion = new_fusion
                print(f"[SET] fusion updated to {current_fusion}\n")
            except Exception:
                print("[ERROR] Invalid fusion. Use weighted, rrf, text, image.\n")
            continue

        results = search_multimodal(
            query=q,
            k=current_k,
            threshold=current_threshold,
            w_text=args.w_text,
            w_img=args.w_img,
            fusion_type=current_fusion,
            rrf_k=args.rrf_k,
        )

        print(f"[SEARCH] Got {len(results)} results.\n")

        if not results:
            continue

        query_name = safe_filename(q)

        save_json_choice = input("Save JSON results for this query? (y/n): ").strip().lower()

        if save_json_choice in {"y", "yes"}:
            results_path = results_dir / f"{query_name}_results.json"
            save_json(results_path, results)

            print(f"[RESULTS] Saved to: {results_path}\n")
        else:
            print("[RESULTS] Not saved.\n")

        save_map = input("Save map for this query? (y/n): ").strip().lower()

        if save_map in {"y", "yes"}:
            map_path = maps_dir / f"{query_name}.html"

            m = build_map(results)

            if m is not None:
                m.save(str(map_path))
                print(f"[MAP] Saved to: {map_path}\n")
        else:
            print("[MAP] Not saved.\n")


if __name__ == "__main__":
    main()