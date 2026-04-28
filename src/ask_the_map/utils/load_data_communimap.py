#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


def load_raw_dataframe(path):
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        print(f"[DATA] Detected Excel file: {path}")
        return pd.read_excel(path)

    if suffix == ".csv":
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception as e1:
            try:
                return pd.read_csv(path, sep=",")
            except Exception as e2:
                try:
                    return pd.read_csv(path, sep=";")
                except Exception as e3:
                    raise ValueError(
                        f"Could not parse CSV.\nAuto: {e1}\nComma: {e2}\nSemicolon: {e3}"
                    )

    raise ValueError(f"Unsupported file type: {suffix}")


def find_media_columns(df):
    """
    Finds image/media columns.

    Handles columns like:
    - IMAGE
    - MEDIA_2635_1
    - MEDIA_2635_2
    - MEDIA_2635_3
    - MEDIA_2635_4
    - MEDIA_2635_5
    """
    media_cols = []

    if "IMAGE" in df.columns:
        media_cols.append("IMAGE")

    for col in df.columns:
        col_str = str(col)
        if col_str.startswith("MEDIA_"):
            media_cols.append(col)

    # remove duplicates while preserving order
    seen = set()
    out = []
    for c in media_cols:
        if c not in seen:
            out.append(c)
            seen.add(c)

    return out


def get_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_communimap_data(path, expand_images=True):
    """
    Load CommuniMap data and standardise it for Ask-the-Map.

    If expand_images=True:
        one CommuniMap record with multiple MEDIA_* images becomes
        multiple rows with the same source_id.

    Returns columns:
        id
        source_id
        image_index
        media_column
        text
        LATITUDE
        LONGITUDE
        primary_image
    """

    print(f"[DATA] Loading CommuniMap data from: {path}")

    df = load_raw_dataframe(path)
    print(f"[DATA] Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} columns")

    lat_col = get_first_existing_column(df, ["LATITUDE", "Latitude", "latitude", "lat"])
    lon_col = get_first_existing_column(
        df,
        ["LONGITUDE", "Longitude", "longitude", "lon", "lng"],
    )
    text_col = get_first_existing_column(
        df,
        ["DESCRIPTION", "Description", "description", "TEXT", "text"],
    )
    id_col = get_first_existing_column(df, ["ID", "id", "source_id", "SOURCE_ID"])

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not find LATITUDE/LONGITUDE columns. "
            f"Available columns: {list(df.columns)}"
        )

    media_cols = find_media_columns(df)

    if not media_cols:
        raise ValueError(
            "No image/media columns found. Expected IMAGE or MEDIA_* columns."
        )

    print("[DATA] Column mapping:")
    print(f"  source_id  <- {id_col if id_col else 'row index'}")
    print(f"  text       <- {text_col if text_col else 'empty string'}")
    print(f"  latitude   <- {lat_col}")
    print(f"  longitude  <- {lon_col}")
    print(f"  media cols <- {len(media_cols)} columns")

    rows = []

    for original_idx, row in df.iterrows():
        lat = pd.to_numeric(row.get(lat_col), errors="coerce")
        lon = pd.to_numeric(row.get(lon_col), errors="coerce")

        if pd.isna(lat) or pd.isna(lon):
            continue

        source_id = (
            str(row.get(id_col)).strip()
            if id_col is not None and pd.notna(row.get(id_col))
            else str(original_idx)
        )

        text = (
            str(row.get(text_col)).strip()
            if text_col is not None and pd.notna(row.get(text_col))
            else ""
        )

        image_values = []

        for media_col in media_cols:
            val = row.get(media_col)

            if isinstance(val, str) and val.strip():
                image_values.append(
                    {
                        "media_column": str(media_col),
                        "primary_image": val.strip(),
                    }
                )

        if not image_values:
            if not expand_images:
                rows.append(
                    {
                        "source_id": source_id,
                        "image_index": 0,
                        "media_column": None,
                        "text": text,
                        "LATITUDE": lat,
                        "LONGITUDE": lon,
                        "primary_image": "",
                    }
                )
            continue

        if expand_images:
            for image_index, image_info in enumerate(image_values):
                rows.append(
                    {
                        "source_id": source_id,
                        "image_index": image_index,
                        "media_column": image_info["media_column"],
                        "text": text,
                        "LATITUDE": lat,
                        "LONGITUDE": lon,
                        "primary_image": image_info["primary_image"],
                    }
                )
        else:
            first = image_values[0]
            rows.append(
                {
                    "source_id": source_id,
                    "image_index": 0,
                    "media_column": first["media_column"],
                    "text": text,
                    "LATITUDE": lat,
                    "LONGITUDE": lon,
                    "primary_image": first["primary_image"],
                }
            )

    clean = pd.DataFrame(rows)

    if clean.empty:
        raise ValueError("No usable CommuniMap rows found after cleaning.")

    clean = clean.reset_index(drop=True)
    clean["id"] = clean.index.astype(str)

    clean = clean[
        [
            "id",
            "source_id",
            "image_index",
            "media_column",
            "text",
            "LATITUDE",
            "LONGITUDE",
            "primary_image",
        ]
    ]

    print(f"[DATA] Cleaned dataset: {len(clean)} image-level rows.")
    print(f"[DATA] Unique CommuniMap records: {clean['source_id'].nunique()}")

    return clean


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_data_communimap.py <path_to_file>")
        sys.exit(1)

    df = load_communimap_data(sys.argv[1], expand_images=True)
    print(df.head(20).to_string())