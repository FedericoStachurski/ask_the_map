#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import folium
from folium.plugins import HeatMap


def build_map(
    results,
    center=None,
    zoom_start=12,
    heat_radius=25,
    heat_blur=30,
):
    if not results:
        print("[MAP] No results provided.")
        return None

    if center is None:
        center = (55.8721, -4.2892)

    m = folium.Map(location=center, zoom_start=zoom_start)

    heat_points = [
        (float(r["lat"]), float(r["lon"]))
        for r in results
        if r.get("lat") is not None and r.get("lon") is not None
    ]

    if heat_points:
        HeatMap(
            heat_points,
            radius=heat_radius,
            blur=heat_blur,
            max_zoom=13,
        ).add_to(m)

    for r in results:
        lat = float(r["lat"])
        lon = float(r["lon"])

        text = (r.get("text") or "")[:220].replace("\n", " ")
        img = r.get("image") or r.get("primary_image") or ""

        score = float(r.get("score", 0.0))
        score_text = float(r.get("score_text", 0.0))
        score_img = float(r.get("score_img", 0.0))
        p_text = float(r.get("p_text", 0.0))
        p_img = float(r.get("p_img", 0.0))

        source_id = r.get("source_id", r.get("id", "unknown"))
        image_index = r.get("image_index", "")
        media_column = r.get("media_column", "")

        html = f"""
        <div style="width:260px;">
            <b>ID:</b> {source_id}<br>
            <b>Image index:</b> {image_index}<br>
            <b>Media column:</b> {media_column}<br>
            <b>Score:</b> {score:.3f}<br>
            <b>Text score:</b> {score_text:.3f} (p: {p_text:.1f}%)<br>
            <b>Image score:</b> {score_img:.3f} (p: {p_img:.1f}%)<br>
            <p style="font-size:11px;">{text}...</p>
            <img src="{img}" width="240">
        </div>
        """

        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            fill=True,
            fill_opacity=0.85,
            color="red",
            popup=folium.Popup(html, max_width=280),
            tooltip=f"ID: {source_id} | score: {score:.3f}",
        ).add_to(m)

    return m


def main():
    parser = argparse.ArgumentParser(description="Build a Folium map from saved results JSON.")

    parser.add_argument(
        "--embedding-folder",
        required=True,
        help="Folder containing saved Ask-the-Map outputs.",
    )

    parser.add_argument(
        "--results-file",
        required=True,
        help="Results JSON filename or path.",
    )

    parser.add_argument(
        "--map-file",
        default=None,
        help="Optional output map filename/path. If omitted, uses results filename.",
    )

    args = parser.parse_args()

    embedding_folder = Path(args.embedding_folder).expanduser().resolve()

    results_path = Path(args.results_file).expanduser()
    if not results_path.is_absolute():
        results_path = embedding_folder / results_path

    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    if args.map_file is None:
        map_path = embedding_folder / f"{results_path.stem}.html"
    else:
        map_path = Path(args.map_file).expanduser()
        if not map_path.is_absolute():
            map_path = embedding_folder / map_path

    map_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    m = build_map(results)

    if m is None:
        return

    m.save(str(map_path))
    print(f"[MAP] Saved to: {map_path}")


if __name__ == "__main__":
    main()