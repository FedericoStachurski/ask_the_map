#!/usr/bin/env python3

import argparse
from sentence_transformers import SentenceTransformer

from ask_the_map.utils.model_registry import MODEL_MAP, get_model_save_path


def list_models() -> None:
    print("\nAvailable models:\n")

    for name, info in MODEL_MAP.items():
        print(f"  {name:18s} [{info['family']}]")
        print(f"    HF: {info['hf_id']}")
        print(f"    {info['description']}\n")


def download_model(model_name: str, force: bool = False, device: str = "cpu") -> None:
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. Use --list to see available options."
        )

    model_id = MODEL_MAP[model_name]["hf_id"]
    save_path = get_model_save_path(model_name)

    if save_path.exists() and not force:
        print(f"[INFO] Model already exists at: {save_path}")
        print("       Use --force to re-download.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] {model_id}")
    print(f"[SAVE]     {save_path}")

    model = SentenceTransformer(model_id, device=device)
    model.save(str(save_path))

    print("[DONE] Model downloaded successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download local vision-language models for Ask the Map"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Model to download. Use --list to see options.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model already exists.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used while downloading/loading model before saving.",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.model_name:
        parser.error("You must provide --model-name or use --list.")

    download_model(
        model_name=args.model_name,
        force=args.force,
        device=args.device,
    )


if __name__ == "__main__":
    main()