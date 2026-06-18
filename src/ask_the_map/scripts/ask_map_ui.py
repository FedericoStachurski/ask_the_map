from pathlib import Path
from datetime import datetime
from io import BytesIO
import argparse
import base64
import html
import json
import mimetypes
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ask_the_map.scripts.ask_pipeline_ui import AskMapPipeline


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--run-dir",
        "--embedding-folder",
        dest="embedding_folder",
        default=None,
        help=(
            "Optional folder containing <folder_name>_text.npy, "
            "<folder_name>_image.npy, <folder_name>_meta.json. "
            "If omitted, the app scans the outputs folder."
        ),
    )

    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Folder containing multiple Ask-the-Map embedding output runs.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--collapse-source-ids",
        action="store_true",
        help="Keep only the best image/result per CommuniMap source_id.",
    )

    args, _ = parser.parse_known_args()
    return args


# =========================================================
# PATHS / STATIC ASSETS
# =========================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

STATIC_DIR = PROJECT_ROOT / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = Path.cwd() / "static"


def find_static_file(patterns):
    for pattern in patterns:
        matches = sorted(STATIC_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


LOGO_PATH = find_static_file(
    [
        "CommuniMap_Logo_Colour*",
        "*CommuniMap*Logo*",
        "*communimap*logo*",
    ]
)

GALLANT_BG_PATH = find_static_file(
    [
        "cropped-gallant-1.webp",
        "cropped-gallant*",
        "*gallant*",
    ]
)

MEDIA_BG_PATH = find_static_file(
    [
        "Media_986283_smx.png",
        "Media_986283*",
        "Media*smx*",
    ]
)


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Ask the Map",
    page_icon="🗺️",
    layout="wide",
)


# =========================================================
# STYLING
# =========================================================

def file_to_data_uri(path):
    if path is None or not Path(path).exists():
        return ""

    path = Path(path)
    mime, _ = mimetypes.guess_type(path)

    if mime is None:
        mime = "image/png"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def inject_css():
    logo_uri = file_to_data_uri(LOGO_PATH)
    gallant_bg_uri = file_to_data_uri(GALLANT_BG_PATH)
    media_bg_uri = file_to_data_uri(MEDIA_BG_PATH)

    # -----------------------------------------------------
    # Main page background
    # -----------------------------------------------------
    if gallant_bg_uri:
        page_bg_css = f"""
        .stApp {{
            background:
                linear-gradient(
                    135deg,
                    rgba(248, 255, 244, 0.965),
                    rgba(232, 250, 255, 0.965)
                ),
                url("{gallant_bg_uri}");
            background-size: min(280px, 18vw) auto;
            background-position: center 5.4rem;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #f7fff4;
        }}
        """
    else:
        page_bg_css = """
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(126, 211, 33, 0.16), transparent 34%),
                radial-gradient(circle at bottom right, rgba(84, 190, 216, 0.20), transparent 38%),
                #f7fff4;
        }
        """

    # -----------------------------------------------------
    # Query card background
    # -----------------------------------------------------
    if media_bg_uri:
        form_bg_css = f"""
        div[data-testid="stForm"] {{
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 255, 255, 0.84),
                    rgba(235, 252, 255, 0.84)
                ),
                url("{media_bg_uri}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            border: 8px solid #111111;
            border-radius: 28px;
            padding: 1.35rem;
            box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22);
            overflow: hidden;
        }}
        """
    else:
        form_bg_css = """
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.92);
            border: 8px solid #111111;
            border-radius: 28px;
            padding: 1.35rem;
            box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22);
            overflow: hidden;
        }
        """

    # -----------------------------------------------------
    # Logo
    # -----------------------------------------------------
    logo_css = ""
    logo_html = ""

    if logo_uri:
        logo_css = """
        .communimap-logo {
            position: fixed;
            top: 4.7rem;
            right: 0.4rem;
            z-index: 9999;
            width: min(250px, 18vw);
            padding: 0.65rem 0.8rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 12px 34px rgba(30, 90, 80, 0.18);
            backdrop-filter: blur(12px);
            border: 4px solid #111111;
        }

        .communimap-logo img {
            width: 100%;
            display: block;
        }

        @media (max-width: 1200px) {
            .communimap-logo {
                position: relative;
                top: auto;
                right: auto;
                width: 240px;
                margin-bottom: 1rem;
            }
        }
        """

        logo_html = f"""
        <div class="communimap-logo">
            <img src="{logo_uri}" alt="CommuniMap logo" />
        </div>
        """

    st.markdown(
        f"""
        <style>
        {page_bg_css}
        {form_bg_css}
        {logo_css}

        :root {{
            --communi-green: #7ed321;
            --communi-blue: #54bed8;
            --soft-green: #eaffdf;
            --soft-blue: #e6faff;
            --deep-green: #063b33;
            --text-dark: #08201d;
            --muted: #4b6464;
        }}

        /* Streamlit top bar */
        header[data-testid="stHeader"],
        div[data-testid="stHeader"],
        [data-testid="stHeader"] {{
            background: linear-gradient(
                90deg,
                rgba(234, 255, 223, 0.98),
                rgba(230, 250, 255, 0.98)
            ) !important;
            box-shadow: 0 1px 0 rgba(84, 190, 216, 0.22) !important;
        }}

        header[data-testid="stHeader"] *,
        [data-testid="stHeader"] *,
        [data-testid="stToolbar"] * {{
            color: var(--deep-green) !important;
        }}

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {{
            background: transparent !important;
        }}

        .block-container {{
            padding-top: 4.9rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: inherit;
        }}

        h1 {{
            color: var(--deep-green) !important;
            font-size: 3.2rem !important;
            font-weight: 850 !important;
            letter-spacing: -0.05em;
            margin-bottom: 0.15rem !important;
        }}

        h2, h3 {{
            color: var(--deep-green) !important;
            letter-spacing: -0.03em;
        }}

        .app-subtitle {{
            color: #315b56;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
            max-width: 720px;
        }}

        .run-chip {{
            display: inline-block;
            background: linear-gradient(90deg, #dfffbd, #dff8ff);
            color: var(--deep-green);
            font-weight: 800;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            margin: 0.4rem 0 1.5rem 0;
            border: 2px solid #111111;
            box-shadow: 0 10px 28px rgba(30, 90, 80, 0.12);
        }}

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    rgba(234, 255, 223, 0.98),
                    rgba(230, 250, 255, 0.98)
                ) !important;
            border-right: 1px solid rgba(84, 190, 216, 0.28);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--deep-green) !important;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: var(--deep-green) !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stAlert"] {{
            background: rgba(84, 190, 216, 0.14);
            border: 3px solid #111111;
            border-radius: 16px;
        }}

        section[data-testid="stSidebar"] .stSelectbox > div > div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.90) !important;
            border-radius: 14px !important;
            border: 3px solid #111111 !important;
            color: var(--deep-green) !important;
        }}

        /* Dropdown menu readability */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="menu"],
        ul[role="listbox"] {{
            background: rgba(255, 255, 255, 0.98) !important;
            color: #063b33 !important;
            border: 3px solid #111111 !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 34px rgba(30, 90, 80, 0.18) !important;
        }}

        li[role="option"],
        div[role="option"] {{
            background: rgba(255, 255, 255, 0.98) !important;
            color: #063b33 !important;
        }}

        li[role="option"] *,
        div[role="option"] * {{
            color: #063b33 !important;
        }}

        li[role="option"]:hover,
        div[role="option"]:hover {{
            background: rgba(223, 248, 255, 0.98) !important;
            color: #063b33 !important;
        }}

        div[data-testid="stForm"],
        div[data-testid="stForm"] *,
        div[data-testid="stForm"] label,
        div[data-testid="stForm"] p,
        div[data-testid="stForm"] span,
        div[data-testid="stForm"] div {{
            color: #062f2a !important;
        }}

        div[data-testid="stForm"] label {{
            color: var(--deep-green) !important;
            font-weight: 800 !important;
        }}

        div[data-testid="stForm"] input,
        div[data-testid="stForm"] textarea,
        .stTextInput input,
        div[data-baseweb="input"] input {{
            background: rgba(255, 255, 255, 0.96) !important;
            border-radius: 14px !important;
            color: #000000 !important;
            border: 1px solid rgba(8, 32, 29, 0.18) !important;
        }}

        div[data-testid="stForm"] input::placeholder,
        div[data-testid="stForm"] textarea::placeholder,
        .stTextInput input::placeholder,
        div[data-baseweb="input"] input::placeholder,
        input::placeholder,
        textarea::placeholder {{
            color: #000000 !important;
            opacity: 0.70 !important;
        }}

        div[data-testid="stForm"] div[data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.96) !important;
            border-radius: 14px !important;
            color: var(--text-dark) !important;
            border: 1px solid rgba(8, 32, 29, 0.12) !important;
        }}

        div[data-testid="stForm"] div[data-baseweb="select"] span {{
            color: var(--text-dark) !important;
        }}

        .stButton > button,
        div[data-testid="stForm"] button {{
            border-radius: 999px !important;
            border: 0 !important;
            padding: 0.65rem 1.15rem !important;
            font-weight: 850 !important;
            color: #05211d !important;
            background: linear-gradient(90deg, var(--communi-green), var(--communi-blue)) !important;
            box-shadow: 0 12px 30px rgba(30, 90, 80, 0.18);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}

        .stButton > button:hover,
        div[data-testid="stForm"] button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 16px 38px rgba(30, 90, 80, 0.24);
        }}

        .saved-map-note {{
            background: rgba(255, 255, 255, 0.94);
            border: 6px solid #111111;
            border-radius: 24px;
            padding: 1rem 1.15rem;
            color: var(--text-dark);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.18);
            margin-bottom: 1rem;
        }}

        .saved-map-note strong {{
            color: var(--text-dark) !important;
        }}

        .metric-pill {{
            display: inline-block;
            background: linear-gradient(90deg, #dfffbd, #dff8ff);
            color: var(--deep-green);
            border: 3px solid #111111;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            font-weight: 750;
            margin-bottom: 0.6rem;
        }}

        .param-card {{
            background: rgba(255, 255, 255, 0.88);
            border: 3px solid #111111;
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
            margin-top: 0.5rem;
            box-shadow: 0 8px 20px rgba(30, 90, 80, 0.08);
        }}

        .param-row {{
            display: flex;
            justify-content: space-between;
            gap: 0.7rem;
            border-bottom: 1px solid rgba(17, 17, 17, 0.18);
            padding: 0.35rem 0;
            font-size: 0.86rem;
        }}

        .param-row:last-child {{
            border-bottom: none;
        }}

        .param-key {{
            font-weight: 750;
            color: var(--deep-green);
        }}

        .param-value {{
            color: #315b56;
            text-align: right;
            word-break: break-word;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 14px 42px rgba(30, 90, 80, 0.14);
        }}

        div[data-testid="stAlert"] {{
            border-radius: 16px;
        }}

        code, pre, kbd,
        section[data-testid="stSidebar"] code,
        section[data-testid="stSidebar"] pre,
        section[data-testid="stSidebar"] kbd {{
            background: rgba(255, 255, 255, 0.76) !important;
            color: #063b33 !important;
            border: 1px solid rgba(84, 190, 216, 0.28) !important;
            border-radius: 8px !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
        }}

        details[data-testid="stExpander"] {{
            background: rgba(255, 255, 255, 0.74) !important;
            border: 1px solid rgba(84, 190, 216, 0.28) !important;
            border-radius: 14px !important;
            overflow: hidden;
        }}

        details[data-testid="stExpander"] summary {{
            background: rgba(255, 255, 255, 0.80) !important;
            color: #063b33 !important;
        }}

        details[data-testid="stExpander"] summary * {{
            color: #063b33 !important;
        }}

        .results-card {{
            background: rgba(255, 255, 255, 0.94);
            border: 8px solid #111111;
            border-radius: 28px;
            padding: 1rem;
            box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22);
            margin-top: 1rem;
            overflow: hidden;
        }}

        .results-scroll {{
            width: 100%;
            max-height: 430px;
            overflow: auto;
            border-radius: 18px;
            border: 1px solid rgba(84, 190, 216, 0.25);
            background: rgba(255, 255, 255, 0.96);
        }}

        table.results-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            color: #062f2a !important;
            background: rgba(255, 255, 255, 0.96);
        }}

        table.results-table thead th {{
            position: sticky;
            top: 0;
            z-index: 2;
            background: linear-gradient(90deg, #eaffdf, #e6faff);
            color: #063b33 !important;
            text-align: left;
            font-weight: 850;
            padding: 0.65rem 0.75rem;
            border-bottom: 1px solid rgba(84, 190, 216, 0.35);
            white-space: nowrap;
        }}

        table.results-table tbody td {{
            color: #062f2a !important;
            padding: 0.55rem 0.75rem;
            border-bottom: 1px solid rgba(84, 190, 216, 0.16);
            vertical-align: top;
        }}

        table.results-table tbody tr:nth-child(even) {{
            background: rgba(230, 250, 255, 0.38);
        }}

        table.results-table tbody tr:nth-child(odd) {{
            background: rgba(255, 255, 255, 0.72);
        }}

        table.results-table tbody tr:hover {{
            background: rgba(223, 255, 189, 0.42);
        }}

        table.results-table td.text-cell {{
            min-width: 360px;
            max-width: 520px;
            white-space: normal;
            line-height: 1.35;
        }}

        table.results-table td.num-cell {{
            text-align: right;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }}

        .scatter-card {{
            background: rgba(255, 255, 255, 0.94);
            border: 8px solid #111111;
            border-radius: 28px;
            padding: 1rem;
            box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22);
            margin-top: 1.2rem;
            overflow: hidden;
        }}

        .scatter-card img {{
            width: 100%;
            display: block;
            border-radius: 16px;
        }}

        .scatter-caption {{
            color: #315b56;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}

        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(84,190,216,0.6), transparent);
            margin: 1.5rem 0;
        }}
        </style>

        {logo_html}
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================================================
# CACHE PIPELINE
# =========================================================

@st.cache_resource
def load_pipeline(embedding_folder, device="auto", collapse_source_ids=False):
    return AskMapPipeline(
        embedding_folder=embedding_folder,
        device=device,
        collapse_source_ids=collapse_source_ids,
    )


# =========================================================
# GENERAL HELPERS
# =========================================================

def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "query"


def map_to_html(fmap):
    if fmap is None:
        return None

    return fmap.get_root().render()


def save_map(saved_dir, query, fmap, results_df, params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(query)

    html_path = saved_dir / f"{timestamp}_{slug}.html"
    meta_path = saved_dir / f"{timestamp}_{slug}.json"
    results_path = saved_dir / f"{timestamp}_{slug}_results.csv"

    html_text = map_to_html(fmap)

    if html_text is None:
        raise ValueError("Cannot save map because the Folium map is None.")

    html_path.write_text(html_text, encoding="utf-8")
    results_df.to_csv(results_path, index=False)

    meta = {
        "query": query,
        "created_at": timestamp,
        "html_file": html_path.name,
        "results_file": results_path.name,
        "params": params,
        "n_results": int(len(results_df)),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return html_path, meta_path, results_path


def load_saved_maps(saved_dir):
    saved = []

    for meta_path in sorted(saved_dir.glob("*.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            html_path = saved_dir / meta["html_file"]

            if html_path.exists():
                saved.append(
                    {
                        "label": f'{meta["created_at"]} | {meta["query"]}',
                        "meta": meta,
                        "html_path": html_path,
                        "meta_path": meta_path,
                    }
                )

        except Exception:
            continue

    return saved


def nice_results_table(results_df):
    preferred_cols = [
        "rank",
        "source_id",
        "score",
        "score_text",
        "score_img",
        "p_text",
        "p_img",
        "text",
        "lat",
        "lon",
        "image_index",
        "media_column",
        "submission_date",
        "CREATED_AT",
        "primary_image",
    ]

    show_cols = [c for c in preferred_cols if c in results_df.columns]

    if show_cols:
        return results_df[show_cols]

    return results_df


def render_results_table(results_df, max_rows=100):
    if results_df is None or results_df.empty:
        st.info("No retrieved results to show.")
        return

    df = nice_results_table(results_df).copy()
    df = df.head(max_rows)

    preferred_order = [
        "rank",
        "source_id",
        "score",
        "score_text",
        "score_img",
        "p_text",
        "p_img",
        "text",
        "lat",
        "lon",
        "image_index",
        "media_column",
        "submission_date",
        "CREATED_AT",
    ]

    cols = [c for c in preferred_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in cols and c != "primary_image"]
    cols = cols + extra_cols

    df = df[cols]

    html_rows = []

    for _, row in df.iterrows():
        cells = []

        for col in df.columns:
            value = row[col]

            if isinstance(value, float):
                value = f"{value:.4g}"

            if value is None:
                value = ""

            value_html = html.escape(str(value))

            if col == "text":
                cell_class = "text-cell"
            elif col in {
                "rank",
                "score",
                "score_text",
                "score_img",
                "p_text",
                "p_img",
                "lat",
                "lon",
                "image_index",
            }:
                cell_class = "num-cell"
            else:
                cell_class = ""

            cells.append(f'<td class="{cell_class}">{value_html}</td>')

        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    header_html = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in df.columns
    )

    table_html = (
        '<div class="results-card">'
        '<div class="results-scroll">'
        '<table class="results-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(html_rows)}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)

    if len(results_df) > max_rows:
        st.caption(
            f"Showing first {max_rows} rows out of {len(results_df)} retrieved results."
        )


def render_score_scatter_matplotlib(results_df):
    if results_df is None or results_df.empty:
        st.info("No score data to plot.")
        return

    if "p_text" not in results_df.columns or "p_img" not in results_df.columns:
        st.info("p_text and p_img are not available for this result set.")
        return

    df = results_df.copy()
    df["p_text"] = pd.to_numeric(df["p_text"], errors="coerce")
    df["p_img"] = pd.to_numeric(df["p_img"], errors="coerce")
    df = df.dropna(subset=["p_text", "p_img"])

    if df.empty:
        st.info("No valid p_text / p_img values to plot.")
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=150)

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.scatter(
        df["p_text"],
        df["p_img"],
        s=60,
        alpha=0.78,
        c="#54bed8",
        edgecolors="#111111",
        linewidths=0.8,
    )

    ax.axvline(
        50,
        color="#7ed321",
        linewidth=1.8,
        linestyle="--",
        alpha=0.55,
    )
    ax.axhline(
        50,
        color="#7ed321",
        linewidth=1.8,
        linestyle="--",
        alpha=0.55,
    )

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    ax.set_xlabel("Text percentile score (p_text)", fontsize=11, color="#063b33")
    ax.set_ylabel("Image percentile score (p_img)", fontsize=11, color="#063b33")
    ax.set_title(
        "Text vs image retrieval signal",
        fontsize=15,
        fontweight="bold",
        color="#063b33",
        pad=14,
    )

    ax.grid(True, color="#54bed8", alpha=0.18, linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.4)

    ax.tick_params(axis="both", colors="#063b33", labelsize=10)

    ax.text(
        98,
        98,
        "Strong text + image",
        ha="right",
        va="top",
        fontsize=9,
        color="#063b33",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#eaffdf",
            edgecolor="#111111",
            linewidth=0.9,
            alpha=0.85,
        ),
    )

    ax.text(
        2,
        98,
        "Image-driven",
        ha="left",
        va="top",
        fontsize=9,
        color="#063b33",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#e6faff",
            edgecolor="#111111",
            linewidth=0.9,
            alpha=0.85,
        ),
    )

    ax.text(
        98,
        2,
        "Text-driven",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#063b33",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#e6faff",
            edgecolor="#111111",
            linewidth=0.9,
            alpha=0.85,
        ),
    )

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    scatter_html = (
        '<div class="scatter-card">'
        f'<img src="data:image/png;base64,{encoded}" alt="p_text versus p_img scatter plot" />'
        '<div class="scatter-caption">'
        "Each point is one retrieved image-level result. "
        "Top-right points are strong in both text and image similarity; "
        "top-left points are more image-driven; bottom-right points are more text-driven."
        "</div>"
        "</div>"
    )

    st.markdown(scatter_html, unsafe_allow_html=True)


def render_map_html(html_text, height=760):
    if html_text is None:
        st.warning("No map HTML to render.")
        return

    escaped_map_html = html.escape(html_text, quote=True)

    bordered_html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                background: transparent;
                overflow: hidden;
                box-sizing: border-box;
            }}

            .map-frame-wrap {{
                width: 100%;
                height: {height}px;
                box-sizing: border-box;
                border: 8px solid #111111;
                border-radius: 26px;
                background: #111111;
                box-shadow: 0 22px 60px rgba(0, 0, 0, 0.28);
                overflow: hidden;
            }}

            .map-frame-wrap iframe {{
                width: 100%;
                height: 100%;
                border: none;
                display: block;
                border-radius: 16px;
                background: white;
            }}
        </style>
    </head>
    <body>
        <div class="map-frame-wrap">
            <iframe srcdoc="{escaped_map_html}"></iframe>
        </div>
    </body>
    </html>
    """

    components.html(
        bordered_html,
        height=height + 20,
        scrolling=False,
    )


def format_param_value(value):
    if isinstance(value, float):
        return f"{value:.3g}"

    if isinstance(value, bool):
        return "true" if value else "false"

    if value is None:
        return "none"

    value = str(value)

    if len(value) > 42:
        return "..." + value[-39:]

    return value


def render_params_card(params, container=None):
    if not params:
        return

    if container is None:
        container = st.sidebar

    rows = []

    for key, value in params.items():
        key_html = html.escape(str(key))
        value_html = html.escape(format_param_value(value))

        rows.append(
            '<div class="param-row">'
            f'<span class="param-key">{key_html}</span>'
            f'<span class="param-value">{value_html}</span>'
            '</div>'
        )

    params_html = (
        '<div class="param-card">'
        + "".join(rows)
        + "</div>"
    )

    container.markdown(
        params_html,
        unsafe_allow_html=True,
    )


# =========================================================
# EMBEDDING DISCOVERY HELPERS
# =========================================================

def find_embedding_files_in_folder(folder: Path):
    folder = Path(folder)
    folder_prefix = folder.name

    preferred_text = folder / f"{folder_prefix}_text.npy"
    preferred_image = folder / f"{folder_prefix}_image.npy"
    preferred_meta = folder / f"{folder_prefix}_meta.json"

    if preferred_text.exists() and preferred_image.exists() and preferred_meta.exists():
        return {
            "valid": True,
            "prefix": folder_prefix,
            "text_path": preferred_text,
            "image_path": preferred_image,
            "meta_path": preferred_meta,
            "reason": None,
        }

    text_files = sorted(folder.glob("*_text.npy"))
    image_files = sorted(folder.glob("*_image.npy"))
    meta_files = sorted(folder.glob("*_meta.json"))

    text_prefixes = {p.name.removesuffix("_text.npy"): p for p in text_files}
    image_prefixes = {p.name.removesuffix("_image.npy"): p for p in image_files}
    meta_prefixes = {p.name.removesuffix("_meta.json"): p for p in meta_files}

    common_prefixes = sorted(
        set(text_prefixes).intersection(image_prefixes).intersection(meta_prefixes)
    )

    if common_prefixes:
        prefix = common_prefixes[0]

        return {
            "valid": True,
            "prefix": prefix,
            "text_path": text_prefixes[prefix],
            "image_path": image_prefixes[prefix],
            "meta_path": meta_prefixes[prefix],
            "reason": None,
        }

    missing = []

    if not text_files:
        missing.append("*_text.npy")
    if not image_files:
        missing.append("*_image.npy")
    if not meta_files:
        missing.append("*_meta.json")

    if missing:
        reason = "missing " + ", ".join(missing)
    else:
        reason = "text/image/meta files exist but prefixes do not match"

    return {
        "valid": False,
        "prefix": None,
        "text_path": None,
        "image_path": None,
        "meta_path": None,
        "reason": reason,
    }


def is_valid_embedding_folder(folder: Path) -> bool:
    return find_embedding_files_in_folder(folder)["valid"]


def get_embedding_folder_summary(folder: Path):
    folder = Path(folder)
    info = find_embedding_files_in_folder(folder)

    if not info["valid"]:
        return {
            "name": folder.name,
            "path": str(folder),
            "prefix": None,
            "text_path": None,
            "image_path": None,
            "meta_path": None,
            "n_text": None,
            "n_image": None,
            "dim_text": None,
            "dim_image": None,
            "n_meta": None,
            "model_name": "unknown",
            "modified": None,
            "valid": False,
            "reason": info["reason"],
        }

    text_path = info["text_path"]
    image_path = info["image_path"]
    meta_path = info["meta_path"]

    summary = {
        "name": folder.name,
        "path": str(folder),
        "prefix": info["prefix"],
        "text_path": str(text_path),
        "image_path": str(image_path),
        "meta_path": str(meta_path),
        "n_text": None,
        "n_image": None,
        "dim_text": None,
        "dim_image": None,
        "n_meta": None,
        "model_name": "unknown",
        "modified": None,
        "valid": True,
        "reason": None,
    }

    try:
        text_arr = np.load(text_path, mmap_mode="r")
        image_arr = np.load(image_path, mmap_mode="r")

        summary["n_text"] = int(text_arr.shape[0])
        summary["n_image"] = int(image_arr.shape[0])
        summary["dim_text"] = int(text_arr.shape[1])
        summary["dim_image"] = int(image_arr.shape[1])

    except Exception:
        pass

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        summary["n_meta"] = len(meta)

        if len(meta) > 0:
            summary["model_name"] = meta[0].get("model_name", "unknown")

    except Exception:
        pass

    try:
        modified_ts = max(
            text_path.stat().st_mtime,
            image_path.stat().st_mtime,
            meta_path.stat().st_mtime,
        )

        summary["modified"] = datetime.fromtimestamp(modified_ts).strftime(
            "%Y-%m-%d %H:%M"
        )

    except Exception:
        pass

    return summary


def discover_embedding_folders(outputs_dir: Path):
    outputs_dir = Path(outputs_dir).expanduser().resolve()

    if not outputs_dir.exists():
        return [], []

    valid = []
    invalid = []

    for folder in sorted(outputs_dir.iterdir()):
        if not folder.is_dir():
            continue

        summary = get_embedding_folder_summary(folder)

        if summary["valid"]:
            valid.append(
                {
                    "folder": folder,
                    "summary": summary,
                }
            )
        else:
            invalid.append(
                {
                    "folder": folder,
                    "summary": summary,
                }
            )

    valid.sort(
        key=lambda x: x["summary"].get("modified") or "",
        reverse=True,
    )

    return valid, invalid


def make_embedding_label(item):
    return item["summary"]["name"]


# =========================================================
# LOAD ARGS
# =========================================================

args = parse_args()

if args.outputs_dir is None:
    outputs_dir = PROJECT_ROOT / "outputs"
else:
    outputs_dir = Path(args.outputs_dir).expanduser().resolve()


# =========================================================
# SIDEBAR: EMBEDDING DATA SELECTION
# =========================================================

st.sidebar.title("Ask-the-Map data")

invalid_embedding_items = []

if args.embedding_folder is not None:
    requested_folder = Path(args.embedding_folder).expanduser().resolve()

    if not is_valid_embedding_folder(requested_folder):
        st.error(
            f"The requested embedding folder is not valid:\n\n{requested_folder}\n\n"
            "Expected a matching triplet:\n"
            "- *_text.npy\n"
            "- *_image.npy\n"
            "- *_meta.json"
        )
        st.stop()

    embedding_items = [
        {
            "folder": requested_folder,
            "summary": get_embedding_folder_summary(requested_folder),
        }
    ]

else:
    embedding_items, invalid_embedding_items = discover_embedding_folders(outputs_dir)

    if not embedding_items:
        st.error(
            f"No valid embedding folders found in:\n\n{outputs_dir}\n\n"
            "A valid folder must contain a matching triplet:\n"
            "- *_text.npy\n"
            "- *_image.npy\n"
            "- *_meta.json"
        )

        if invalid_embedding_items:
            with st.expander("Skipped folders"):
                for item in invalid_embedding_items:
                    st.write(
                        f"**{item['summary']['name']}** — {item['summary']['reason']}"
                    )

        st.stop()


embedding_labels = [make_embedding_label(item) for item in embedding_items]

selected_embedding_label = st.sidebar.selectbox(
    "Embedding run",
    embedding_labels,
    index=0,
)

selected_item = embedding_items[embedding_labels.index(selected_embedding_label)]
embedding_folder = selected_item["folder"]
embedding_summary = selected_item["summary"]

st.sidebar.markdown("### Selected data")

selected_data_params = {
    "Folder": embedding_summary["name"],
    "Model": embedding_summary.get("model_name", "unknown"),
    "Metadata rows": embedding_summary.get("n_meta", "unknown"),
    "Text": (
        f"{embedding_summary.get('n_text', '?')} × {embedding_summary.get('dim_text', '?')}"
    ),
    "Image": (
        f"{embedding_summary.get('n_image', '?')} × {embedding_summary.get('dim_image', '?')}"
    ),
    "Modified": embedding_summary.get("modified", "unknown"),
}

if embedding_summary.get("prefix") and embedding_summary["prefix"] != embedding_summary["name"]:
    selected_data_params["File prefix"] = embedding_summary["prefix"]

render_params_card(selected_data_params)

with st.sidebar.expander("Embedding paths"):
    st.write(embedding_summary["path"])
    st.write(embedding_summary["text_path"])
    st.write(embedding_summary["image_path"])
    st.write(embedding_summary["meta_path"])

if invalid_embedding_items:
    with st.sidebar.expander("Skipped output folders"):
        for item in invalid_embedding_items:
            st.write(
                f"**{item['summary']['name']}** — {item['summary']['reason']}"
            )


# =========================================================
# CLEAR STATE IF EMBEDDING FOLDER CHANGES
# =========================================================

previous_embedding_folder = st.session_state.get("active_embedding_folder")

if previous_embedding_folder != str(embedding_folder):
    for key in [
        "latest_query",
        "latest_results_df",
        "latest_results",
        "latest_map",
        "latest_map_html",
        "latest_params",
    ]:
        st.session_state.pop(key, None)

    st.session_state["active_embedding_folder"] = str(embedding_folder)


# =========================================================
# LOAD PIPELINE
# =========================================================

saved_dir = embedding_folder / "saved_ui_maps"
saved_dir.mkdir(parents=True, exist_ok=True)

with st.spinner("Loading Ask-the-Map model and FAISS indices..."):
    pipeline = load_pipeline(
        embedding_folder=str(embedding_folder),
        device=args.device,
        collapse_source_ids=args.collapse_source_ids,
    )


# =========================================================
# SIDEBAR: SAVED MAPS
# =========================================================

st.sidebar.markdown("---")
st.sidebar.title("Saved maps")
st.sidebar.caption(f"Run folder: `{embedding_folder.name}`")

saved_maps = load_saved_maps(saved_dir)
selected_saved = None

if saved_maps:
    labels = ["None - current query"] + [m["label"] for m in saved_maps]
    selected_label = st.sidebar.selectbox("Open saved map", labels)

    if selected_label != "None - current query":
        selected_saved = next(m for m in saved_maps if m["label"] == selected_label)

        st.sidebar.markdown("### Saved query")
        st.sidebar.write(selected_saved["meta"]["query"])
        st.sidebar.write(f'Results: {selected_saved["meta"]["n_results"]}')

        if "params" in selected_saved["meta"]:
            st.sidebar.markdown("### Parameters")
            render_params_card(selected_saved["meta"]["params"])

else:
    st.sidebar.info("No saved maps yet.")


# =========================================================
# MAIN APP
# =========================================================

st.title("Ask the Map")

st.markdown(
    """
    <div class="app-subtitle">
        Search CommuniMap observations in plain English and visualise the retrieved points on a local interactive map.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="run-chip">
        Embedding folder: {html.escape(embedding_folder.name)}
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SHOW SAVED MAP
# =========================================================

if selected_saved is not None:
    st.subheader("Saved map")

    st.markdown(
        f"""
        <div class="saved-map-note">
            <strong>Query:</strong> {html.escape(str(selected_saved["meta"]["query"]))}<br>
            <strong>Results:</strong> {html.escape(str(selected_saved["meta"]["n_results"]))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    html_text = selected_saved["html_path"].read_text(encoding="utf-8")

    render_map_html(
        html_text,
        height=760,
    )


# =========================================================
# CURRENT QUERY MODE
# =========================================================

else:
    with st.form("query_form"):
        query = st.text_input(
            "Query",
            placeholder=(
                "e.g. flooding in streets near schools, "
                "urban trees, unsafe crossings"
            ),
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            top_k = st.slider(
                "Top K",
                min_value=10,
                max_value=5000,
                value=100,
                step=10,
            )

        with col2:
            threshold = st.slider(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )

        with col3:
            fusion_type = st.selectbox(
                "Fusion",
                ["weighted", "rrf", "text", "image"],
                index=0,
            )

        col4, col5, col6 = st.columns(3)

        with col4:
            text_weight = st.slider(
                "Text weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
            )

        with col5:
            image_weight = st.slider(
                "Image weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
            )

        with col6:
            rrf_k = st.number_input(
                "RRF k",
                min_value=1,
                max_value=200,
                value=60,
                step=1,
            )

        search_clicked = st.form_submit_button("Search map")

    if fusion_type == "rrf":
        st.info(
            "RRF scores are much smaller than weighted/text/image scores, "
            "so the search threshold is automatically set to 0.00 for RRF."
        )

    if search_clicked:
        if not query.strip():
            st.warning("Please type a query first.")

        else:
            effective_threshold = threshold

            if fusion_type == "rrf":
                effective_threshold = 0.0

            with st.spinner("Searching the map..."):
                results_df, fmap, results = pipeline.run(
                    query=query,
                    k=top_k,
                    threshold=effective_threshold,
                    w_text=text_weight,
                    w_img=image_weight,
                    fusion_type=fusion_type,
                    rrf_k=rrf_k,
                )

                map_html = map_to_html(fmap)

            st.session_state["latest_query"] = query
            st.session_state["latest_results_df"] = results_df
            st.session_state["latest_results"] = results
            st.session_state["latest_map"] = fmap
            st.session_state["latest_map_html"] = map_html
            st.session_state["latest_params"] = {
                "top_k": top_k,
                "threshold": effective_threshold,
                "ui_threshold": threshold,
                "fusion_type": fusion_type,
                "text_weight": text_weight,
                "image_weight": image_weight,
                "rrf_k": rrf_k,
                "collapse_source_ids": args.collapse_source_ids,
                "embedding_folder": str(embedding_folder),
            }

    if "latest_map" in st.session_state:
        st.subheader("Current map")

        latest_map = st.session_state["latest_map"]
        latest_map_html = st.session_state.get("latest_map_html")
        results_df = st.session_state["latest_results_df"]

        if latest_map is None or latest_map_html is None or results_df.empty:
            st.warning("No results found for this query.")

        else:
            render_map_html(
                latest_map_html,
                height=760,
            )

            col_a, col_b = st.columns([1, 4])

            with col_a:
                if st.button("Save this map"):
                    html_path, meta_path, results_path = save_map(
                        saved_dir=saved_dir,
                        query=st.session_state["latest_query"],
                        fmap=latest_map,
                        results_df=results_df,
                        params=st.session_state["latest_params"],
                    )

                    st.success(f"Saved map: {html_path.name}")
                    st.info(f"Saved results: {results_path.name}")
                    st.rerun()

            st.subheader("Retrieved results")

            st.markdown(
                f"""
                <div class="metric-pill">
                    Number of results: {len(results_df)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_results_table(results_df, max_rows=100)

            st.subheader("Retrieval signal scatter")

            render_score_scatter_matplotlib(results_df)

    else:
        st.info("Type a query and click **Search map**.")