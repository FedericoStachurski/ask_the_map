"""
Ask the Map — FastAPI Application

Backend:
    - Existing AskMapPipeline
    - CLIP / embedding model
    - FAISS text and image retrieval
    - Folium map generation

Frontend:
    - HTML / CSS / JavaScript
    - Original Streamlit styling
"""

import json
import os

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ask_the_map.scripts.ask_pipeline_ui import AskMapPipeline


# =========================================================
# PATHS
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

STATIC_DIR = ROOT_DIR / "static"
FRONTEND_DIR = STATIC_DIR / "fastapi"


# =========================================================
# EMBEDDING CONFIGURATION
# =========================================================

def get_embedding_folder():

    configured_path = os.getenv("ASK_MAP_EMBEDDING_DIR")

    if not configured_path:
        raise RuntimeError(
            "ASK_MAP_EMBEDDING_DIR is not configured. "
            "Set it to the folder containing the "
            "precomputed Ask the Map embeddings."
        )

    embedding_folder = Path(
        configured_path
    ).expanduser().resolve()

    if not embedding_folder.is_dir():
        raise RuntimeError(
            f"Embedding folder does not exist: {embedding_folder}"
        )

    return embedding_folder


# =========================================================
# LOAD PIPELINE ONCE
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    embedding_folder = get_embedding_folder()

    print("[FASTAPI] Loading Ask the Map pipeline...")

    app.state.pipeline = AskMapPipeline(
        embedding_folder=embedding_folder,
        device=os.getenv("ASK_MAP_DEVICE", "cpu"),
        collapse_source_ids=False,
    )

    print("[FASTAPI] Pipeline ready!")

    yield

    app.state.pipeline = None

    print("[FASTAPI] Application stopped.")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Ask the Map",
    description="Multimodal retrieval over CommuniMap observations",
    version="0.2.0",
    lifespan=lifespan,
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-store"}
    )


# =========================================================
# REQUEST MODEL
# =========================================================

class SearchRequest(BaseModel):

    query: str = Field(min_length=1, max_length=1000)

    top_k: int = Field(default=100, ge=1, le=5000)

    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    fusion_type: str = Field(
        default="weighted",
        pattern="^(weighted|rrf|text|image)$",
    )

    text_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
    )

    image_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health(request: Request):

    pipeline = request.app.state.pipeline

    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "model_name": pipeline.model_name,
        "observations": len(pipeline.meta),
    }

# =========================================================
# DATASET INFORMATION
# =========================================================

@app.get("/api/info")
def dataset_info(request: Request):

    pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Search pipeline is not loaded.",
        )

    return {
        "application": "Ask the Map",
        "embedding_run": pipeline.embedding_folder.name,
        "model_name": pipeline.model_name,
        "observations": len(pipeline.meta),
        "text_embeddings": int(pipeline.index_text.ntotal),
        "image_embeddings": int(pipeline.index_img.ntotal),
        "embedding_dimension": int(pipeline.index_text.d),
        "device": pipeline.device,
        "search_enabled": True,
    }

# =========================================================
# SEARCH
# =========================================================

@app.post("/api/search")
def search_map(
    payload: SearchRequest,
    request: Request,
):

    pipeline = request.app.state.pipeline

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Search pipeline is not ready.",
        )

    effective_threshold = (
        0.0
        if payload.fusion_type == "rrf"
        else payload.threshold
    )

    try:

        results_df, fmap, results = pipeline.run(
            query=payload.query,
            k=payload.top_k,
            threshold=effective_threshold,
            w_text=payload.text_weight,
            w_img=payload.image_weight,
            fusion_type=payload.fusion_type,
            rrf_k=payload.rrf_k,
        )

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Search could not be completed.",
        ) from exc

    # Convert Folium map into HTML
    map_html = (
        fmap.get_root().render()
        if fmap is not None
        else None
    )

    # Ensure dataframe values are JSON serialisable
    results_json = json.loads(
        results_df.to_json(orient="records")
    )

    return {
        "query": payload.query,
        "n_results": len(results_json),
        "results": results_json,
        "map_html": map_html,
    }
