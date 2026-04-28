from pathlib import Path


MODEL_MAP = {
    "clip-b32": {
        "hf_id": "sentence-transformers/clip-ViT-B-32",
        "family": "clip",
        "description": "CLIP ViT-B/32. Good default: lighter and fast.",
    },
    "clip-b16": {
        "hf_id": "sentence-transformers/clip-ViT-B-16",
        "family": "clip",
        "description": "CLIP ViT-B/16. Better image detail, heavier than B/32.",
    },
    "clip-l14": {
        "hf_id": "sentence-transformers/clip-ViT-L-14",
        "family": "clip",
        "description": "CLIP ViT-L/14. Stronger but much heavier.",
    },
    "siglip-b16-224": {
        "hf_id": "sentence-transformers/google-siglip-base-patch16-224",
        "family": "siglip",
        "description": "Google SigLIP base patch16 224.",
    },
    "siglip-b16-384": {
        "hf_id": "sentence-transformers/google-siglip-base-patch16-384",
        "family": "siglip",
        "description": "Higher-resolution SigLIP. Better detail, slower.",
    },
    "siglip2-b16-224": {
        "hf_id": "sentence-transformers/google-siglip2-base-patch16-224",
        "family": "siglip2",
        "description": "SigLIP2 base 224. Newer SigLIP-style model.",
    },
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_models_dir() -> Path:
    return get_project_root() / "models"


def get_model_save_path(model_name: str) -> Path:
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}")

    family = MODEL_MAP[model_name]["family"]
    return get_models_dir() / family / model_name