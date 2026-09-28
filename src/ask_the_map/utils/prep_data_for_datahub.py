#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import pandas as pd

from ask_the_map.utils.load_data_communimap import (
    load_raw_dataframe,
)


# ============================================================
# Exact columns to remove from PUBLIC dataset
# ============================================================

PUBLIC_PRIVATE_COLUMNS = {
    "ID",
    "ROOT_ID",
    "USER_ID",
    "AUDIO_RECORDING",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "SIGNATURE",
}


def find_media_columns(df):
    """
    Find columns containing image/media URLs.

    Private dataset:
        These columns are retained.

    Public dataset:
        These columns are removed.

    Handles examples such as:
        IMAGE
        MEDIA_2635_0
        MEDIA_2635_1
        IMAGE_MEDIA_0
        IMAGE_MEDIA_1
        PHOTO
    """

    media_cols = []

    exact_media_columns = {
        "IMAGE",
        "IMAGES",
        "PHOTO",
        "PHOTOS",
        "IMAGE_URL",
        "IMAGE_URLS",
        "PHOTO_URL",
        "PHOTO_URLS",
    }

    for col in df.columns:

        name = str(col).strip()
        upper = name.upper()

        if (
            upper in exact_media_columns
            or upper.startswith("MEDIA_")
            or upper.startswith("IMAGE_MEDIA_")
            or upper.startswith("PHOTO_MEDIA_")
        ):
            media_cols.append(col)

    return list(dict.fromkeys(media_cols))

    # Remove duplicates while preserving column order
    return list(dict.fromkeys(media_cols))


def prepare_data_for_datahub(
    input_path,
    output_dir,
    name,
):

    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. LOAD ORIGINAL COMMUNIMAP DATA
    # ========================================================

    print("\n======================================")
    print("LOADING COMMUNIMAP DATA")
    print("======================================")

    print(f"\nInput file:\n{input_path}")

    df = load_raw_dataframe(input_path)

    # Strip whitespace from column names only.
    # Do NOT alter values.
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    original_rows = len(df)
    original_columns = list(df.columns)

    print(
        f"\nOriginal dataset:"
        f"\n  Rows:    {original_rows}"
        f"\n  Columns: {len(original_columns)}"
    )

    # ========================================================
    # 2. CHECK FOR DUPLICATE COLUMN NAMES
    # ========================================================

    duplicate_columns = (
        pd.Series(original_columns)
        .value_counts()
    )

    duplicate_columns = duplicate_columns[
        duplicate_columns > 1
    ]

    if not duplicate_columns.empty:

        raise ValueError(
            "Duplicate column names detected after loading:\n"
            + duplicate_columns.to_string()
        )

    # ========================================================
    # 3. FIND MEDIA / IMAGE URL COLUMNS
    # ========================================================

    media_columns = find_media_columns(df)

    print(
        f"\nDetected {len(media_columns)} "
        f"image/media column(s):"
    )

    for col in media_columns:
        print(f"  - {col}")

    # ========================================================
    # 4. FIND PRIVATE IDENTIFIER COLUMNS
    # ========================================================

    identifier_columns = []

    for col in df.columns:

        if str(col).strip().upper() in PUBLIC_PRIVATE_COLUMNS:
            identifier_columns.append(col)

    print(
        f"\nDetected {len(identifier_columns)} "
        f"identifier/private column(s):"
    )

    for col in identifier_columns:
        print(f"  - {col}")

    # ========================================================
    # 5. PRIVATE DATASET
    # ========================================================

    # IMPORTANT:
    # This is a complete copy of the loaded CommuniMap data.
    # NOTHING is removed.
    private_df = df.copy()

    private_path = (
        output_dir
        / f"communimap_private_{name}.csv"
    )

    # Explicitly write comma-separated CSV
    private_df.to_csv(
        private_path,
        index=False,
        sep=",",
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    # ========================================================
    # 6. PUBLIC DATASET
    # ========================================================

    # Remove ONLY identifiers + image/media URL columns
    columns_to_remove = list(
        dict.fromkeys(
            identifier_columns
            + media_columns
        )
    )

    public_df = df.drop(
        columns=columns_to_remove,
        errors="raise",
    ).copy()

    public_path = (
        output_dir
        / f"communimap_public_{name}.csv"
    )

    # Explicitly write comma-separated CSV
    public_df.to_csv(
        public_path,
        index=False,
        sep=",",
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    # ========================================================
    # 7. VALIDATION BEFORE FINISHING
    # ========================================================

    print("\n======================================")
    print("VALIDATING EXPORT")
    print("======================================")

    # --------------------------------------------------------
    # Private must contain every original column
    # --------------------------------------------------------

    if list(private_df.columns) != original_columns:
        raise RuntimeError(
            "PRIVATE export column mismatch. "
            "Some original columns were changed or lost."
        )

    if len(private_df) != original_rows:
        raise RuntimeError(
            "PRIVATE export row count differs "
            "from original dataset."
        )

    # --------------------------------------------------------
    # Public should differ ONLY by specified removed columns
    # --------------------------------------------------------

    expected_public_columns = [
        col
        for col in original_columns
        if col not in columns_to_remove
    ]

    if list(public_df.columns) != expected_public_columns:
        raise RuntimeError(
            "PUBLIC export contains an unexpected "
            "column difference."
        )

    if len(public_df) != original_rows:
        raise RuntimeError(
            "PUBLIC export row count differs "
            "from original dataset."
        )

    # ========================================================
    # 8. READ FILES BACK FROM DISK
    # ========================================================

    # This confirms that the actual files written to disk
    # can be read correctly as comma-separated CSV files.

    private_check = pd.read_csv(
        private_path,
        sep=",",
        dtype=str,
        keep_default_na=False,
    )

    public_check = pd.read_csv(
        public_path,
        sep=",",
        dtype=str,
        keep_default_na=False,
    )

    # --------------------------------------------------------
    # Check PRIVATE disk file
    # --------------------------------------------------------

    if list(private_check.columns) != original_columns:
        raise RuntimeError(
            "PRIVATE CSV failed read-back validation. "
            "Columns on disk do not match original columns."
        )

    if len(private_check) != original_rows:
        raise RuntimeError(
            "PRIVATE CSV failed read-back validation. "
            "Row count differs."
        )

    # --------------------------------------------------------
    # Check PUBLIC disk file
    # --------------------------------------------------------

    if list(public_check.columns) != expected_public_columns:
        raise RuntimeError(
            "PUBLIC CSV failed read-back validation. "
            "Unexpected columns found."
        )

    if len(public_check) != original_rows:
        raise RuntimeError(
            "PUBLIC CSV failed read-back validation. "
            "Row count differs."
        )

    # ========================================================
    # 9. FINAL REPORT
    # ========================================================

    print("\n======================================")
    print("DATAHUB EXPORT COMPLETE")
    print("======================================")

    print("\nPRIVATE DATASET")
    print(f"  File:    {private_path}")
    print(f"  Rows:    {len(private_df)}")
    print(f"  Columns: {len(private_df.columns)}")

    print("\nPUBLIC DATASET")
    print(f"  File:    {public_path}")
    print(f"  Rows:    {len(public_df)}")
    print(f"  Columns: {len(public_df.columns)}")

    print("\nColumns removed ONLY from public dataset:")

    for col in columns_to_remove:
        print(f"  - {col}")

    print("\nValidation:")
    print("  ✓ Private row count matches original")
    print("  ✓ Private contains every original column")
    print("  ✓ Public row count matches original")
    print("  ✓ Public removed only requested columns")
    print("  ✓ Both outputs are comma-separated")
    print("  ✓ Both files successfully read back from disk")

    print("\n[DATAHUB] Done.")


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare private and public CommuniMap "
            "datasets for the GALLANT DataHub."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to original CommuniMap CSV/XLSX.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name, e.g. Sept2026.",
    )

    parser.add_argument(
        "--output-dir",
        default="./data/datahub_data",
        help="Output directory.",
    )

    args = parser.parse_args()

    prepare_data_for_datahub(
        input_path=args.input,
        output_dir=args.output_dir,
        name=args.name,
    )


if __name__ == "__main__":
    main()