// =========================================================
// ASK THE MAP — FASTAPI FRONTEND
// =========================================================


// =========================================================
// LOAD DATASET INFORMATION
// =========================================================

async function loadDatasetInfo() {

    try {

        const response = await fetch("/api/info");

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Update sidebar information
        const fields = {
            "embedding-run": data.embedding_run,
            "model-name": data.model_name,
            "metadata-rows": data.observations.toLocaleString(),
            "text-embeddings": data.text_embeddings.toLocaleString(),
            "image-embeddings": data.image_embeddings.toLocaleString(),
            "active-run": data.embedding_run
        };

        for (const [id, value] of Object.entries(fields)) {

            const element = document.getElementById(id);

            if (element) {
                element.textContent = value;
            }
        }

        // Update embedding dropdown
        const selector = document.getElementById("embedding-selector");

        if (selector) {

            selector.replaceChildren(
                new Option(data.embedding_run, data.embedding_run)
            );

            // Dataset switching is not implemented yet
            selector.disabled = true;
        }

        console.log("[ASK MAP] Dataset loaded:", data);

    } catch (error) {

        console.error(
            "[ASK MAP] Unable to load dataset information:",
            error
        );
    }
}


// =========================================================
// RENDER RETRIEVED RESULTS
// =========================================================

function renderResultsTable(results) {

    const container = document.getElementById("results");

    container.replaceChildren();

    if (!results || results.length === 0) {
        container.textContent = "No observations found.";
        return;
    }

    const columns = [
        ["rank", "Rank"],
        ["source_id", "Source ID"],
        ["score", "Score"],
        ["score_text", "Text score"],
        ["score_img", "Image score"],
        ["text", "Observation"],
        ["lat", "Latitude"],
        ["lon", "Longitude"]
    ];

    const scroll = document.createElement("div");
    scroll.className = "results-scroll";

    const table = document.createElement("table");
    table.className = "results-table";

    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    for (const [, label] of columns) {
        const th = document.createElement("th");
        th.textContent = label;
        headerRow.appendChild(th);
    }

    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    for (const result of results) {

        const row = document.createElement("tr");

        for (const [key] of columns) {

            const td = document.createElement("td");

            let value = result[key] ?? "";

            if (typeof value === "number" && !Number.isInteger(value)) {
                value = value.toFixed(4);
            }

            td.textContent = value;

            if (key === "text") {
                td.className = "text-cell";
            }

            row.appendChild(td);
        }

        tbody.appendChild(row);
    }

    table.appendChild(tbody);
    scroll.appendChild(table);
    container.appendChild(scroll);
}

// =========================================================
// RENDER RETRIEVAL SCATTER PLOT
// =========================================================

function renderRetrievalScatter(results) {

    const container = document.getElementById("scatter-container");

    if (!container) {
        return;
    }

    if (!Array.isArray(results) || results.length === 0) {
        container.textContent = "No retrieval scores available.";
        return;
    }

    // Keep only observations with valid numeric scores.
    const valid = results.filter(result =>
        result.score_text !== null &&
        result.score_text !== undefined &&
        result.score_img !== null &&
        result.score_img !== undefined &&
        Number.isFinite(Number(result.score_text)) &&
        Number.isFinite(Number(result.score_img))
    );

    if (valid.length === 0) {
        container.textContent = "No valid text/image score pairs found.";
        return;
    }

    if (typeof Plotly === "undefined") {
        container.textContent =
            "Plotting library unavailable. Check your internet connection.";
        return;
    }

    const trace = {

        x: valid.map(result => Number(result.score_text)),
        y: valid.map(result => Number(result.score_img)),

        text: valid.map(result =>
            `Source ID: ${result.source_id ?? "Unknown"}`
        ),

        customdata: valid.map(result => [
            result.rank ?? "",
            String(result.source_id ?? ""),
            String(result.text ?? "").slice(0, 250)
        ]),

        mode: "markers",
        type: "scattergl",

        marker: {
            size: 10,
            color: valid.map(result => Number(result.score ?? 0)),
            colorscale: [
                [0, "#d9f3c7"],
                [0.5, "#7ed321"],
                [1, "#54bed8"]
            ],
            showscale: true,
            colorbar: {
                title: "Score"
            },
            opacity: 0.8,
            line: {
                color: "#063b33",
                width: 0.5
            }
        },

        hovertemplate:
            "<b>Source %{customdata[1]}</b><br>" +
            "Rank: %{customdata[0]}<br>" +
            "Text score: %{x:.4f}<br>" +
            "Image score: %{y:.4f}<br>" +
            "%{customdata[2]}" +
            "<extra></extra>"
    };

    const layout = {

        title: {
            text: "Text vs image retrieval",
            font: {
                size: 20,
                color: "#063b33"
            }
        },

        xaxis: {
            title: "Text similarity score",
            zeroline: false,
            gridcolor: "#e5ede9"
        },

        yaxis: {
            title: "Image similarity score",
            zeroline: false,
            gridcolor: "#e5ede9"
        },

        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(255,255,255,0.85)",

        margin: {
            l: 75,
            r: 75,
            t: 65,
            b: 75
        },

        hovermode: "closest",
        autosize: true
    };

    Plotly.react(
        container,
        [trace],
        layout,
        {
            responsive: true,
            displaylogo: false
        }
    );
}



// =========================================================
// PERFORM SEARCH
// =========================================================

async function searchMap() {

    const query = document.getElementById("query").value.trim();
    const status = document.getElementById("status");
    const resultCount = document.getElementById("result-count");
    const mapContainer = document.getElementById("map-container");
    const resultsContainer = document.getElementById("results");


    const searchButton = document.querySelector(
        '#query-form button[type="submit"]'
    );

    if (!query) {

        resultsContainer.textContent = "Please enter a query.";
        return;
    }

    // Collect search parameters
    const payload = {
        query: query,
        top_k: Number(document.getElementById("top-k").value),
        threshold: Number(document.getElementById("threshold").value),
        fusion_type: document.getElementById("fusion").value,
        text_weight: Number(document.getElementById("text-weight").value),
        image_weight: Number(document.getElementById("image-weight").value),
        rrf_k: Number(document.getElementById("rrf-k").value)
    };

    console.log("[ASK MAP] Sending search:", payload);

    status.textContent = "Searching CommuniMap...";

    resultCount.textContent = "0";

    resultsContainer.innerHTML = `
        <p class="muted">Retrieving observations...</p>
    `;

    if (searchButton) {
        searchButton.disabled = true;
        searchButton.textContent = "Searching...";
    }

    try {

        const response = await fetch("/api/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {

            const error = await response.json();

            throw new Error(
                typeof error.detail === "string"
                    ? error.detail
                    : `HTTP ${response.status}`
            );
        }

        const data = await response.json();

        console.log("[ASK MAP] Search completed:", data.n_results);

        // Update status
        status.textContent = "Search completed successfully!";

        // Update result count
        resultCount.textContent = data.n_results.toLocaleString();

        // Update result count
        resultCount.textContent = data.n_results.toLocaleString();

        // Display retrieved observations
        renderResultsTable(data.results);

        // Display retrieval scatter plot
        renderRetrievalScatter(data.results);

        // Display generated Folium map
        if (data.map_html) {

            const iframe = document.createElement("iframe");

            iframe.className = "results-map";
            iframe.title = "Ask the Map search results";

            iframe.srcdoc = data.map_html;

            mapContainer.replaceChildren(iframe);

        } else {

            mapContainer.textContent =
                "No matching observations found.";
        }

    } catch (error) {

        console.error("[ASK MAP] Search error:", error);

        status.textContent = `Search failed: ${error.message}`;

    } finally {

        if (searchButton) {
            searchButton.disabled = false;
            searchButton.textContent = "Search map";
        }
    }
}


// =========================================================
// INITIALISE INTERFACE
// =========================================================

document.addEventListener("DOMContentLoaded", () => {

    console.log("[ASK MAP] Frontend initialised");

    // Load embedding dataset information
    loadDatasetInfo();

    // Connect search form
    const queryForm = document.getElementById("query-form");

    if (queryForm) {

        queryForm.addEventListener("submit", async (event) => {

            // Prevent browser refresh
            event.preventDefault();

            await searchMap();

        });

    } else {

        console.error("[ASK MAP] Query form not found.");

    }

    // =========================================================
    // LIVE SLIDER VALUES
    // =========================================================

    const sliders = [
        ["top-k", "top-k-value", 0],
        ["threshold", "threshold-value", 2],
        ["text-weight", "text-weight-value", 2],
        ["image-weight", "image-weight-value", 2]
    ];

    for (const [inputId, outputId, decimals] of sliders) {

        const input = document.getElementById(inputId);
        const output = document.getElementById(outputId);

        if (!input || !output) {
            continue;
        }

        function updateValue() {
            output.textContent = Number(input.value).toFixed(decimals);
        }

        input.addEventListener("input", updateValue);

        updateValue();
    }

});
