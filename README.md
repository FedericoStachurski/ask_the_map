<p align="center">
  <img src="static/CommuniMap_Logo_Colour_horizontal_5000px.png" width="360"/>
  &nbsp;&nbsp;&nbsp;
  <img src="static/Media_986283_smxx.png" width="110"/>
</p>

# Ask the Map

Multimodal search over geolocated citizen science data, built for CommuniMap.

The tool takes CommuniMap records with text, coordinates, and images, embeds the text and images into a shared CLIP/SigLIP vector space, and lets the user search the dataset using natural language. Results can be visualised on an interactive Folium map and optionally evaluated using weak validation labels such as trees or flooding.

---

## Features

- Text + image embedding using CLIP / SigLIP-style models  
- Local model download and loading  
- Multimodal search using FAISS  
- Fusion strategies: `weighted`, `text`, `image`, and `rrf`  
- Interactive map generation with Folium  
- Weak validation for trees and flooding  
- Retrieval metrics: Precision@K, Recall@K, NDCG, AP, and R~  

---

## Structure

```text
ask_the_map/
├── src/
│   └── ask_the_map/
│       ├── scripts/
│       │   ├── build_embeddings.py
│       │   ├── search_interactive.py
│       │   ├── make_map.py
│       │   └── download_models.py
│       │
│       └── utils/
│           ├── load_data_communimap.py
│           ├── retrieval_metrics.py
│           ├── model_registry.py
│           └── load_model.py
│
├── static/
├── models/
├── outputs/
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate ask_the_map_env
pip install -e .
```

CLI tools available:

```bash
ask-map-build
ask-map-search
ask-map-map
ask-map-download-model
```

---

## Download a Model

```bash
ask-map-download-model --list
ask-map-download-model --model-name clip-b32
```

Models are stored in:

```text
models/clip/clip-b32/
```

---

## Build Embeddings

```bash
ask-map-build <path_to_communimap_file> <output_folder_name> --model-name clip-b32 --device cuda
```

### Example

```bash
ask-map-build \
"/path/to/communiMap data.xlsx" \
March2026_CLIP \
--model-name clip-b32 \
--device cuda
```

Outputs:

```text
outputs/March2026_CLIP/
├── March2026_CLIP_text.npy
├── March2026_CLIP_image.npy
└── March2026_CLIP_meta.json
```

### What happens in this step

- The CommuniMap spreadsheet is loaded  
- Relevant columns are identified (ID, DESCRIPTION, LATITUDE, LONGITUDE, MEDIA_*)  
- Records with multiple images are expanded into image-level rows  
- Text descriptions are embedded  
- Images are embedded  
- Metadata is saved for traceability  

---

## Search Interactively

```bash
ask-map-search \
--embedding-folder outputs/March2026_CLIP \
--data-path "/path/to/communimap.xlsx"
```

### Example queries

```text
trees near roads
flooding in streets
standing water
litter near river
```

### Commands inside the interactive search

```text
k=500
threshold=0.01
fusion=rrf
fusion=weighted
fusion=text
fusion=image
VALIDATE
exit
```

---

## Fusion Modes

```text
weighted  → combines text + image similarity  
text      → text only  
image     → image only  
rrf       → reciprocal rank fusion  
```

---

## Saving Results and Maps

After each query, the tool prompts to save outputs.

```text
outputs/<embedding_folder>/results/
outputs/<embedding_folder>/maps/
```

Example:

```text
outputs/March2026_CLIP/results/flooding_in_streets_results.json
outputs/March2026_CLIP/maps/flooding_in_streets.html
```

---

## Build Map from Saved Results

```bash
ask-map-map \
--embedding-folder outputs/March2026_CLIP \
--results-file results/flooding_in_streets_results.json \
--map-file flooding_map.html
```

---

## Validation

Inside interactive search:

```text
VALIDATE
```

Choose:

~~~text
1) trees
2) flooding / puddles / standing water
~~~

Validation uses weak labels from CommuniMap data:

- `TREE = yes`  
- `TYPE_OF_WATER_EVENT = flooding / puddle / standing water`  

Outputs:

```text
outputs/<embedding_folder>/validation/
```

Includes:

```text
*_summary.json
*_summary.csv
*_ranked_ids.csv
*_relevant_ids.csv
*_curves.png
*_curves_data.csv
*_rank_hist.png
*_rank_hist_data.csv
*_cumulative.png
*_cumulative_data.csv
```

---

## Metrics

- Precision@K  
- Recall@K  
- Hits@K  
- NDCG@K  
- Average Precision (AP)  
- NDCG  
- R~ / Rank*  

Important:

```text
R~ = normalized average rank metric
https://link.springer.com/chapter/10.1007/3-540-45479-9_5
```

---

## Git Notes

Do NOT commit:

```text
models/
outputs/
*.egg-info/
__pycache__/
*.pyc
```

Commit:

```text
src/
README.md
pyproject.toml
```

---

## Typical Workflow

```bash
conda activate ask_the_map_env

pip install -e .

ask-map-download-model --model-name clip-b32

ask-map-build \
"/path/to/communimap.xlsx" \
March2026_CLIP \
--model-name clip-b32 \
--device cuda

ask-map-search \
--embedding-folder outputs/March2026_CLIP \
--data-path "/path/to/communimap.xlsx"
```