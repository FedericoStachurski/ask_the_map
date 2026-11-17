import os
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer

from ask_the_map.utils.model_registry import (
    MODEL_MAP,
    get_model_save_path,
)


# =========================================================
# RESOLVE DEVICE
# =========================================================

def resolve_device(device: str = "auto") -> str:

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but no GPU is available."
        )

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "device must be one of: auto, cpu, cuda"
        )

    return device


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(
    model_name: str,
    device: str = "auto"
) -> SentenceTransformer:

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}")

    resolved_device = resolve_device(device)

    # Optional model directory override.
    configured_path = os.getenv("ASK_MAP_MODEL_DIR")

    if configured_path:

        local_path = Path(configured_path).expanduser()

        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path

        local_path = local_path.resolve()

        # An explicitly configured model must exist.
        # Avoid silently downloading a different model.
        if not local_path.is_dir():
            raise FileNotFoundError(
                f"Configured CLIP model directory not found: "
                f"{local_path}"
            )

    else:

        # Preserve existing behaviour.
        local_path = get_model_save_path(model_name)

    # -----------------------------------------------------
    # LOAD LOCAL MODEL
    # -----------------------------------------------------

    if local_path.is_dir():

        print(f"[MODEL] Loading local model: {local_path}")
        print(f"[DEVICE] {resolved_device}")

        return SentenceTransformer(
            str(local_path),
            device=resolved_device
        )

    # -----------------------------------------------------
    # FALLBACK TO HUGGING FACE
    # -----------------------------------------------------

    hf_id = MODEL_MAP[model_name]["hf_id"]

    print(f"[MODEL] Loading from HuggingFace: {hf_id}")
    print(f"[DEVICE] {resolved_device}")

    return SentenceTransformer(
        hf_id,
        device=resolved_device
    )