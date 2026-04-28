#!/usr/bin/env python3

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from ask_the_map.utils.load_data_communimap import load_communimap_data
from ask_the_map.utils.model_registry import MODEL_MAP, get_model_save_path


DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"


# =========================================================
# DEVICE
# =========================================================
def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no GPU is available.")

    return device


# =========================================================
# MODEL UTILS
# =========================================================
def list_available_models():
    print("\nAvailable registered models:\n")
    for name, info in MODEL_MAP.items():
        path = get_model_save_path(name)
        status = "downloaded" if path.exists() else "missing"
        print(f"  {name:18s} [{info['family']}] - {status}")
        print(f"    {info['description']}")
        print(f"    path: {path}\n")


def prompt_for_model():
    list_available_models()

    while True:
        model_name = input("Choose model name: ").strip()

        if model_name not in MODEL_MAP:
            print(f"[ERROR] Unknown model: {model_name}")
            continue

        model_path = get_model_save_path(model_name)

        if not model_path.exists():
            print(f"\n[ERROR] Model not downloaded:")
            print(f"  {model_path}")
            print(f"\nRun:")
            print(f"  ask-map-download-model --model-name {model_name}\n")
            continue

        return model_name


def require_local_model(model_name: str) -> Path:
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model_name}'.")

    model_path = get_model_save_path(model_name)

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModel missing:\n  {model_path}\n"
            f"\nDownload with:\n  ask-map-download-model --model-name {model_name}\n"
        )

    return model_path


# =========================================================
# ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multimodal embeddings for CommuniMap."
    )

    parser.add_argument("data_path")
    parser.add_argument("prefix_name", nargs="?", default=None)

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--list-models", action="store_true")

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--blip-model", default=DEFAULT_BLIP_MODEL)

    parser.add_argument("--batch-size-blip", type=int, default=128)
    parser.add_argument("--batch-size-text", type=int, default=128)
    parser.add_argument("--batch-size-img", type=int, default=128)
    parser.add_argument("--min-text-len", type=int, default=0)
    parser.add_argument("--max-blip-tokens", type=int, default=20)

    return parser.parse_args()


# =========================================================
# IMAGE
# =========================================================
def download_image(url, timeout=7):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# =========================================================
# BLIP
# =========================================================
def load_blip(model_spec: str, device: str):
    processor = BlipProcessor.from_pretrained(model_spec)
    model = BlipForConditionalGeneration.from_pretrained(model_spec).to(device)
    model.eval()
    return processor, model


def caption_batch_blip(images, processor, model, device, max_new_tokens):
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return [processor.decode(seq, skip_special_tokens=True).strip() for seq in out]


def fill_short_descriptions_with_blip(df, model_spec, device, min_len, batch_size, max_tokens):
    norm_text = df["text"].fillna("").astype(str).str.strip()
    needs = (norm_text.str.len() < min_len) & df["primary_image"].notna()

    subset = df[needs].copy()

    if subset.empty:
        print("[BLIP] No captioning needed.")
        return df

    first = subset.sort_values(["source_id", "image_index"]).groupby("source_id").first().reset_index()

    processor, model = load_blip(model_spec, device)

    for i in tqdm(range(0, len(first), batch_size), desc="[BLIP]"):
        batch = first.iloc[i:i + batch_size]

        imgs = []
        ids = []

        for _, row in batch.iterrows():
            img = download_image(row["primary_image"])
            if img:
                imgs.append(img)
                ids.append(row["source_id"])

        if not imgs:
            continue

        captions = caption_batch_blip(imgs, processor, model, device, max_tokens)

        for sid, cap in zip(ids, captions):
            df.loc[df["source_id"] == sid, "text"] = cap

    return df


# =========================================================
# EMBEDDING
# =========================================================
def load_embedding_model(model_name, device):
    path = require_local_model(model_name)
    return SentenceTransformer(str(path), device=device)


def embed_texts(texts, model, batch):
    vecs = []
    for i in tqdm(range(0, len(texts), batch), desc="[TEXT]"):
        v = model.encode(texts[i:i+batch], convert_to_numpy=True, normalize_embeddings=True)
        vecs.append(v.astype(np.float32))
    return np.vstack(vecs)


def embed_images(urls, model, batch):
    n = len(urls)
    out = None

    for i in tqdm(range(0, n, batch), desc="[IMG]"):
        imgs, idxs = [], []

        for j, url in enumerate(urls[i:i+batch]):
            img = download_image(url)
            if img:
                imgs.append(img)
                idxs.append(i + j)

        if not imgs:
            continue

        v = model.encode(imgs, convert_to_numpy=True, normalize_embeddings=True)

        if out is None:
            out = np.zeros((n, v.shape[1]), dtype=np.float32)

        for k, idx in enumerate(idxs):
            out[idx] = v[k]

    return out if out is not None else np.zeros((n, 1), dtype=np.float32)


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()

    if args.list_models:
        list_available_models()
        return

    device = resolve_device(args.device)

    model_name = args.model_name or prompt_for_model()
    require_local_model(model_name)

    data_path = Path(args.data_path).expanduser().resolve()

    outputs_root = Path("outputs")
    prefix_name = args.prefix_name or f"{data_path.stem}_{model_name}"

    output_dir = outputs_root / prefix_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = output_dir / prefix_name

    print("\n[CONFIG]")
    print(f"Data path:     {data_path}")
    print(f"Output folder: {output_dir}")
    print(f"Model:         {model_name}")
    print(f"Device:        {device}")

    df = load_communimap_data(str(data_path))
    print(f"[DATA] Loaded {len(df)} rows")

    df = fill_short_descriptions_with_blip(
        df,
        args.blip_model,
        device,
        args.min_text_len,
        args.batch_size_blip,
        args.max_blip_tokens,
    )

    texts = [" ".join(str(t).split()[:50]) for t in df["text"]]
    imgs = df["primary_image"].tolist()

    model = load_embedding_model(model_name, device)

    text_vecs = embed_texts(texts, model, args.batch_size_text)
    img_vecs = embed_images(imgs, model, args.batch_size_img)

    np.save(str(output_prefix) + "_text.npy", text_vecs)
    np.save(str(output_prefix) + "_image.npy", img_vecs)

    meta = [
        {
            "id": str(i),
            "source_id": str(df.loc[i, "source_id"]),
            "image_index": int(df.loc[i, "image_index"]),
            "media_column": str(df.loc[i, "media_column"]),
            "text": texts[i],
            "primary_image": imgs[i],
            "lat": df.loc[i, "LATITUDE"],
            "lon": df.loc[i, "LONGITUDE"],
        }
        for i in range(len(df))
    ]

    with open(str(output_prefix) + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[DONE]")
    print(f"Saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()