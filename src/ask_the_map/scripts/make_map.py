#!/usr/bin/env python3

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import folium
from branca.element import Element, MacroElement, Template
from folium.plugins import HeatMap


# =========================================================
# SAFE HELPERS
# =========================================================
def safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def normalize_submission_date(value):
    """
    Convert a date value to clean ISO format: YYYY-MM-DD.

    Handles:
        2026-04-24
        2026-04-24T10:58:00
        24/04/2026 10:58
        24/04/2026
    """
    if value is None:
        return None

    s = str(value).strip()

    if not s:
        return None

    # Already ISO-like: 2026-04-24 or 2026-04-24T10:58:00
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]

    # Common UK / CommuniMap style dates
    formats = [
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date().isoformat()
        except Exception:
            pass

    return None


def get_submission_date(r):
    """
    Prefer the clean ISO date from meta/results JSON.
    Fallback to CREATED_AT if needed.
    """
    raw_value = (
        r.get("submission_date")
        or r.get("CREATED_AT")
        or r.get("created_at")
        or r.get("date")
        or None
    )

    return normalize_submission_date(raw_value)


def pretty_date(date_value):
    """
    Display date in popup.
    If the value is YYYY-MM-DD, show it as DD Mon YYYY.
    """
    date_iso = normalize_submission_date(date_value)

    if not date_iso:
        return "Unknown"

    year, month, day = date_iso.split("-")

    month_names = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "Jun",
        "07": "Jul",
        "08": "Aug",
        "09": "Sep",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }

    return f"{day} {month_names.get(month, month)} {year}"


# =========================================================
# TIME SLIDER CONTROL
# =========================================================
def add_time_filter_control(m, marker_records, heat_layer=None):
    """
    Add a month-by-month timeline slider to filter markers by submission date.

    Timeline:
        Jan 2025 -> current month

    Modes:
        - Selected month
        - Up to month
        - All
    """
    if not marker_records:
        return

    map_name = m.get_name()
    heat_name = heat_layer.get_name() if heat_layer is not None else "null"

    marker_data_js = ",\n".join(
        [
            (
                "{"
                f"marker: {rec['marker_name']}, "
                f"lat: {rec['lat']:.8f}, "
                f"lon: {rec['lon']:.8f}, "
                f"submission_date: {json.dumps(rec.get('submission_date'))}"
                "}"
            )
            for rec in marker_records
        ]
    )

    css = """
    <style>
        .atm-time-filter {
            background: white;
            padding: 12px 14px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.30);
            font-family: Arial, sans-serif;
            font-size: 12px;
            width: 320px;
            z-index: 9999;
        }

        .atm-time-filter-title {
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 13px;
        }

        .atm-timeline-label {
            font-weight: bold;
            color: #2b6cb0;
            margin-bottom: 5px;
            font-size: 13px;
        }

        .atm-range {
            width: 100%;
            margin-top: 4px;
            margin-bottom: 4px;
        }

        .atm-slider-endpoints {
            display: flex;
            justify-content: space-between;
            color: #666;
            font-size: 10px;
            margin-bottom: 8px;
        }

        .atm-filter-row {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 7px;
        }

        .atm-filter-btn {
            border: 1px solid #999;
            background: #f7f7f7;
            border-radius: 4px;
            padding: 4px 7px;
            cursor: pointer;
            font-size: 11px;
        }

        .atm-filter-btn:hover {
            background: #e6e6e6;
        }

        .atm-filter-btn.active {
            background: #2b6cb0;
            color: white;
            border-color: #2b6cb0;
        }

        .atm-filter-count {
            margin-top: 8px;
            color: #555;
            font-size: 11px;
        }

        .atm-filter-note {
            margin-top: 5px;
            color: #777;
            font-size: 10px;
            line-height: 1.25;
        }
    </style>
    """

    m.get_root().header.add_child(Element(css))

    template = Template(
        f"""
        {{% macro script(this, kwargs) %}}

        (function() {{
            const map = {map_name};
            const heatLayer = {heat_name};

            const markerData = [
                {marker_data_js}
            ];

            let currentMode = "up_to";

            function pad2(x) {{
                return String(x).padStart(2, "0");
            }}

            function buildMonthList() {{
                const months = [];

                // Start timeline at January 2025
                let d = new Date(2025, 0, 1);

                // End timeline at the current month
                const now = new Date();
                const end = new Date(now.getFullYear(), now.getMonth(), 1);

                while (d <= end) {{
                    const year = d.getFullYear();
                    const month = d.getMonth() + 1;

                    const ym = year + "-" + pad2(month);

                    const label = d.toLocaleString(
                        "en-GB",
                        {{
                            month: "short",
                            year: "numeric"
                        }}
                    );

                    months.push({{
                        ym: ym,
                        label: label
                    }});

                    d.setMonth(d.getMonth() + 1);
                }}

                return months;
            }}

            const monthList = buildMonthList();

            function getItemMonth(item) {{
                if (!item.submission_date) {{
                    return null;
                }}

                const s = String(item.submission_date);

                // Expected:
                // 2026-04-24
                // or 2026-04-24T10:58:00
                if (s.length >= 7 && s[4] === "-") {{
                    return s.slice(0, 7);
                }}

                return null;
            }}

            function shouldShowMarker(item, selectedMonth) {{
                if (currentMode === "all") {{
                    return true;
                }}

                const itemMonth = getItemMonth(item);

                // Undated entries only appear in All mode
                if (!itemMonth) {{
                    return false;
                }}

                if (currentMode === "month") {{
                    return itemMonth === selectedMonth;
                }}

                if (currentMode === "up_to") {{
                    return itemMonth <= selectedMonth;
                }}

                return true;
            }}

            function setActiveButton() {{
                const buttons = document.querySelectorAll(".atm-filter-btn");

                buttons.forEach(function(button) {{
                    const mode = button.getAttribute("data-mode");

                    if (mode === currentMode) {{
                        button.classList.add("active");
                    }} else {{
                        button.classList.remove("active");
                    }}
                }});
            }}

            function updateMonthLabel(slider) {{
                const monthLabel = document.getElementById("atm-current-month-label");

                if (!monthLabel || !monthList.length) {{
                    return;
                }}

                const idx = Number(slider.value);
                const monthInfo = monthList[idx];

                if (monthInfo) {{
                    monthLabel.textContent = monthInfo.label;
                }}
            }}

            function applyTimelineFilter() {{
                const slider = document.getElementById("atm-month-slider");

                if (!slider || !monthList.length) {{
                    return;
                }}

                const selectedIndex = Number(slider.value);
                const selectedMonth = monthList[selectedIndex].ym;

                updateMonthLabel(slider);

                let visibleCount = 0;
                const heatPoints = [];

                markerData.forEach(function(item) {{
                    const keep = shouldShowMarker(item, selectedMonth);

                    if (keep) {{
                        if (!map.hasLayer(item.marker)) {{
                            item.marker.addTo(map);
                        }}

                        visibleCount += 1;
                        heatPoints.push([item.lat, item.lon]);
                    }} else {{
                        if (map.hasLayer(item.marker)) {{
                            map.removeLayer(item.marker);
                        }}
                    }}
                }});

                if (heatLayer && typeof heatLayer.setLatLngs === "function") {{
                    heatLayer.setLatLngs(heatPoints);
                }}

                const countBox = document.getElementById("atm-filter-count");

                if (countBox) {{
                    let modeText = "";

                    if (currentMode === "all") {{
                        modeText = "All entries";
                    }} else if (currentMode === "month") {{
                        modeText = "Selected month";
                    }} else if (currentMode === "up_to") {{
                        modeText = "Up to selected month";
                    }}

                    countBox.textContent =
                        modeText + ": " + visibleCount + " / " + markerData.length + " entries shown";
                }}

                setActiveButton();
            }}

            const control = L.control({{ position: "topright" }});

            control.onAdd = function() {{
                const div = L.DomUtil.create("div", "atm-time-filter");

                const maxSliderValue = Math.max(0, monthList.length - 1);
                const startLabel = monthList.length ? monthList[0].label : "Jan 2025";
                const endLabel = monthList.length ? monthList[monthList.length - 1].label : "Today";

                div.innerHTML = `
                    <div class="atm-time-filter-title">Submission timeline</div>

                    <div id="atm-current-month-label" class="atm-timeline-label">
                        ${{endLabel}}
                    </div>

                    <input
                        id="atm-month-slider"
                        class="atm-range"
                        type="range"
                        min="0"
                        max="${{maxSliderValue}}"
                        value="${{maxSliderValue}}"
                        step="1"
                    >

                    <div class="atm-slider-endpoints">
                        <span>${{startLabel}}</span>
                        <span>${{endLabel}}</span>
                    </div>

                    <div class="atm-filter-row">
                        <button class="atm-filter-btn" data-mode="month">Selected month</button>
                        <button class="atm-filter-btn" data-mode="up_to">Up to month</button>
                        <button class="atm-filter-btn" data-mode="all">All</button>
                    </div>

                    <div id="atm-filter-count" class="atm-filter-count"></div>

                    <div class="atm-filter-note">
                        Drag the slider month by month from Jan 2025 to the current month.
                        Undated entries appear only under All.
                    </div>
                `;

                L.DomEvent.disableClickPropagation(div);
                L.DomEvent.disableScrollPropagation(div);

                const slider = div.querySelector("#atm-month-slider");

                slider.addEventListener("input", function() {{
                    applyTimelineFilter();
                }});

                div.addEventListener("click", function(event) {{
                    if (!event.target.classList.contains("atm-filter-btn")) {{
                        return;
                    }}

                    currentMode = event.target.getAttribute("data-mode");
                    applyTimelineFilter();
                }});

                return div;
            }};

            control.addTo(map);

            // Initial state: cumulative view up to current month
            currentMode = "up_to";
            applyTimelineFilter();

        }})();

        {{% endmacro %}}
        """
    )

    macro = MacroElement()
    macro._template = template
    m.add_child(macro)


# =========================================================
# MAP BUILDER
# =========================================================
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

    valid_results = []

    for r in results:
        lat = safe_float(r.get("lat"))
        lon = safe_float(r.get("lon"))

        if lat is None or lon is None:
            continue

        valid_results.append((r, lat, lon))

    if not valid_results:
        print("[MAP] No valid lat/lon values found.")
        return None

    heat_points = [
        (lat, lon)
        for _, lat, lon in valid_results
    ]

    heat_layer = None

    if heat_points:
        heat_layer = HeatMap(
            heat_points,
            radius=heat_radius,
            blur=heat_blur,
            max_zoom=13,
        )
        heat_layer.add_to(m)

    marker_records = []

    for r, lat, lon in valid_results:
        text = (r.get("text") or "")[:220].replace("\n", " ")
        text = html.escape(text)

        img = r.get("image") or r.get("primary_image") or ""
        img = html.escape(str(img), quote=True)

        score = safe_float(r.get("score"), 0.0)
        score_text = safe_float(r.get("score_text"), 0.0)
        score_img = safe_float(r.get("score_img"), 0.0)
        p_text = safe_float(r.get("p_text"), 0.0)
        p_img = safe_float(r.get("p_img"), 0.0)

        source_id = html.escape(str(r.get("source_id", r.get("id", "unknown"))))
        image_index = html.escape(str(r.get("image_index", "")))
        media_column = html.escape(str(r.get("media_column", "")))

        submission_date = get_submission_date(r)
        submission_display = html.escape(pretty_date(submission_date))

        img_html = ""
        if img:
            img_html = f'<img src="{img}" width="240">'

        popup_html = f"""
        <div style="width:260px;">
            <b>ID:</b> {source_id}<br>
            <b>Submitted:</b> {submission_display}<br>
            <b>Image index:</b> {image_index}<br>
            <b>Media column:</b> {media_column}<br>
            <b>Score:</b> {score:.3f}<br>
            <b>Text score:</b> {score_text:.3f} (p: {p_text:.1f}%)<br>
            <b>Image score:</b> {score_img:.3f} (p: {p_img:.1f}%)<br>
            <p style="font-size:11px;">{text}...</p>
            {img_html}
        </div>
        """

        marker = folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            fill=True,
            fill_opacity=0.85,
            color="red",
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"ID: {source_id} | score: {score:.3f} | submitted: {submission_display}",
        )

        marker.add_to(m)

        marker_records.append(
            {
                "marker_name": marker.get_name(),
                "lat": lat,
                "lon": lon,
                "submission_date": submission_date,
            }
        )

    add_time_filter_control(
        m=m,
        marker_records=marker_records,
        heat_layer=heat_layer,
    )

    return m


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build a Folium map from saved Ask-the-Map results JSON."
    )

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