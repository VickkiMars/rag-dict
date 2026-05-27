# RAG Dictionary & Observable Retrieval System
### FastAPI, FAISS, Sentence-Transformers, Apache Airflow, Redis, Prometheus, Grafana, Docker Compose

A self-contained, privacy-focused semantic lookup dictionary supporting forward (word ➔ meaning) and reverse (meaning ➔ word) queries using Sentence-BERT embeddings, FAISS vector indexing, Redis caching, and full production observability.

---

## 1. System Architecture & Component Workings

The system is decoupled into two primary pipelines: the **Offline Ingestion & Embedding Pipeline** (orchestrated by Apache Airflow) and the **Online Search & Serving Pipeline** (served by FastAPI with Redis caching).

```mermaid
flowchart TD
    subgraph Offline Ingestion & Ingest Pipeline (Airflow)
        A[(dictionary.json)] -->|1. Read & Validate| B(Airflow DAG)
        B -->|2. Generate Embeddings| C(Sentence-BERT)
        B -->|3. Compile Prefix Trie| D(Marisa Trie)
        B -->|4. Populate Cache| E[(Redis Cache)]
        C -->|5. Build Index| F(FAISS flat-IP)
        D -->|Save to disk| G(dictionary.trie)
        F -->|Save to disk| H(embeddings.faiss)
      end

    subgraph Online Search & Serving Pipeline (FastAPI)
        U[Client Web UI] -->|Query| API(FastAPI Server)
        API -->|1. Hit/Miss Check| E
        API -.->|2. Trie Lookup / Suggestions| G
        API -.->|3. Vector Search| H
        API -->|4. Lexical Rerank| BM25(BM25 Okapi)
        API -->|5. Expose Metrics| Prom(Prometheus Scraper)
    end

    subgraph Observability Stack
        Prom -->|Scrape /metrics| TSDB[(Prometheus)]
        Grafana(Grafana Dashboard) -->|Query| TSDB
    end
```

---

## 2. Deep Dive: Key Stack Components

### 🌀 Apache Airflow: The Ingestion Orchestrator
Airflow handles the heavy lifting of parsing, validating, embedding, and loading the vector index and cache. By moving this workflow to Airflow, we decouple resource-intensive model inference from the runtime serving API:
* **Decoupled Workflow Tasks**: The DAG contains distinct validation, building (FAISS + Trie), and warming (Redis) steps.
* **XCom Optimization**: Passing large datasets (like a 305MB JSON file or large Numpy arrays) through Airflow's metadata database (XCom) causes performance issues and database bloat. Our DAG optimizes this by having tasks write locally to the shared host volume `/app/data/` and passing only small metadata records (e.g., records count) via XCom.
* **Single Image Setup**: The DAG mounts our custom base image, meaning all dependencies (`sentence-transformers`, `faiss-cpu`, `marisa-trie`) are packaged once and reused across all tasks.

### ⚡ Redis: High-Performance Caching Layer
Redis is deployed to reduce lookup latencies to **sub-millisecond (<15ms)** times.
* **Forward Caching (`rag_dict:define:<word>`)**: The full word definitions, POS tags, and etymologies are cached upon the first lookup (and pre-warmed by Airflow).
* **Reverse Caching (`rag_dict:reverse:<phrase>`)**: The semantic result lists (including calculated similarity scores and BM25 scores) are cached.
* **Metric Logging**: If a query is present in Redis, the API bypasses FAISS lookup and model inference entirely, instantly returning the value and incrementing the `rag_cache_hit_total` metric.

### 🔥 Prometheus: Metrics Collection
FastAPI exposes real-time runtime metrics via the `prometheus-fastapi-instrumentator` package on the `/metrics` endpoint. Prometheus scrapes this endpoint every 15 seconds:
* **API Health & Traffic**: Tracks total HTTP requests, error rates (4xx/5xx status codes), and query counts.
* **Latency Percentiles**: Measures P50, P90, and P99 latency buckets for all search endpoints.
* **Custom Performance Indicators**:
  * `rag_cache_hit_total`: Total queries resolved directly by the Redis cache.
  * `rag_cache_miss_total`: Total cache misses requiring database or vector lookup.
  * `rag_vector_lookup_total`: Count of semantic searches performed on the FAISS flat index.

### 📈 Grafana: The Visualization Layer
#### Is Grafana Really Necessary? What for?
**Yes, Grafana is absolutely essential.** While Prometheus acts as the database (storing the raw time-series metrics), it only provides a rudimentary graph utility. Grafana acts as the visualization and operational intelligence layer:
1. **Interactive dashboards**: Grafana compiles Prometheus metrics into elegant visual dashboards, making it easy to track P95 latency trends, request volumes, and cache hit ratios.
2. **Alerting**: Grafana can monitor Prometheus metrics and trigger alerts (via Slack, Email, etc.) if latency percentiles spike above **15ms** or error rates increase.
3. **Pre-provisioned Access**: Grafana is fully provisioned in our `docker-compose.yml` to boot with the Prometheus datasource and our custom metrics dashboard preloaded. It also has anonymous login enabled, so you can access the dashboard immediately without login credentials!

---

## 3. Directory & File Relationships

```
.
├── docker-compose.yml              # Configures the multi-container stack (FastAPI, Redis, Airflow, Prometheus, Grafana)
├── Dockerfile.fastapi              # Build recipe for the shared python container
├── requirements.txt                # Python package list
├── prometheus.yml                  # Configures Prometheus scraper targets
├── datasource.yml                  # Configures Grafana default Prometheus connection
│
├── data/                           # Shared volume directory
│   ├── dictionary.json             # Raw input word dictionary (305MB)
│   ├── dictionary.trie             # Compiled binary marisa-trie (for fast prefix autocomplete)
│   └── embeddings.faiss            # Flat IP FAISS vector index of all definitions
│
├── airflow_project/
│   └── dags/
│       └── rag_dictionary_dag.py   # Airflow workflow DAG (Index Builder & Redis Cache Warmer)
│
├── grafana_provisioning/
│   └── dashboards/
│       ├── dashboard_provider.yml  # Registers custom dashboards with Grafana
│       └── dashboard.json          # RAG metric panels definition
│
└── src/                            # FastAPI microservice source
    ├── serve.py                    # Entry point, mounts static UI, handles /define, /reverse, /autocomplete
    ├── config.py                   # Environment and system variables
    ├── cache.py                    # Redis cache operations (JSON & base64 numpy encoders)
    ├── embeddings.py               # HuggingFace SentenceTransformers wrapper
    ├── vector_store.py             # FAISS indexing and BM25 helpers
    ├── reranker.py                 # Lexical-semantic re-scoring algorithm
    ├── dict_trie.py                # Trie compiler and search utils
    ├── monitor.py                  # Custom Prometheus counters definition
    └── frontend/                   # UI Assets
        ├── index.html              # Gorgeous Single-page UI
        ├── style.css               # Modern glassmorphism dark-mode style
        └── app.js                  # Frontend interactive queries & latency timer
```

---

## 4. Setup & Running the Stack

Make sure you have **Docker** and **Docker Compose** installed.

### Step 1: Clone and Start the Services
Launch the entire containerized infrastructure:
```bash
docker compose up --build -d
```
This will:
1. Build the unified base image once and spin up `rag-fastapi`.
2. Spin up `rag-redis`.
3. Launch `rag-airflow-init` to initialize the database schema in `data/airflow.db` and create the admin credentials.
4. Launch `rag-airflow-webserver` and `rag-airflow-scheduler`.
5. Launch `rag-prometheus` and `rag-grafana`.

### Step 2: Trigger the Ingestion DAG
Open your browser and navigate to the Airflow Webserver:
* **URL**: [http://localhost:8080](http://localhost:8080)
* **Credentials**: Username: `admin` | Password: `admin`

Locate the DAG **`rag_dictionary_dag`** and click the **Trigger DAG** button in the upper-right corner. This will run the offline ingestion pipeline:
1. Validate the local dictionary.
2. Build the case-insensitive `dictionary.trie` file.
3. Encode all definitions using HuggingFace sentence embeddings and compile the `embeddings.faiss` flat index.
4. Warm the Redis cache with the full definition payloads.

### Step 3: Use the Dictionary UI
Navigate to the web interface:
* **URL**: [http://localhost:8000](http://localhost:8000)

* **Forward Lookup (Word ➔ Meaning)**: Type a word in the search box. Autocomplete dropdown list items will update in real time. Click search or select a suggestion to pull definitions. Latency will be displayed in green (<15ms).
* **Reverse Lookup (Meaning ➔ Word)**: Enter a phrase describing a concept (e.g. "a device that shows time"). The system will embed the phrase, query FAISS, rerank results with BM25, and present the matching terms alongside their calculated scores.

### Step 4: Monitor Observability Dashboards
To view live metrics under load, navigate to Grafana:
* **URL**: [http://localhost:3000](http://localhost:3000)
* **Credentials**: Automatically logged in anonymously! (Admin dashboard control active).

Click on the **LexiSeek RAG Observability Dashboard** to see live stats, including QPS rate, Redis hit percentages, vector lookup indicators, and real-time response latency percentiles.

---

## 5. Terminal CLI Utility (`define`)

The system comes with a highly intuitive, premium terminal CLI tool that automatically switches query types based on the input:
* **Single Word Input**: Triggers an exact, case-insensitive forward dictionary definition lookup (`/define`).
* **Multi-Word Input**: Triggers a Sentence-BERT + FAISS vector lookup and lexical BM25 reranking (`/reverse`).

### Setup

To make the command available system-wide in your terminal, choose one of these options:

#### Option A: Create a Shell Alias (Recommended)
Add this alias to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`):
```bash
alias define='/home/kami/Desktop/codebase/rag-dict/define'
```
Reload your terminal profile:
```bash
source ~/.bashrc
```

#### Option B: Symlink to `/usr/local/bin`
```bash
sudo ln -s /home/kami/Desktop/codebase/rag-dict/define /usr/local/bin/define
```

### Usage Examples

1. **Exact Word Lookup** (single word argument):
   ```bash
   define serendipity
   ```
   *Output:* Beautifully colored terminal POS badges, phonetics, lists of definitions, etymology, and API transaction latency.
   
2. **Spelling Suggestions** (misspelled word):
   ```bash
   define serendpity
   ```
   *Output:* Automatically runs autocomplete routines and proposes cased spelling alternatives.

3. **Semantic Lookup** (multi-word description or sentence):
   ```bash
   define "an unexpected happy discovery"
   ```
   *Output:* Lists the top semantic terms sorted by combined lexical-semantic scores, alongside detailed vector cosine similarities.

---

## 6. Manual Endpoint Verification

If you prefer testing endpoints using `curl` or tools like Postman, execute these commands:

#### 1. Forward Lookup (Define)
```bash
curl "http://localhost:8000/define?word=serendipity"
```

#### 2. Reverse Lookup (Semantic Phrase Search)
```bash
curl "http://localhost:8000/reverse?meaning=an%20unexpected%20happy%20discovery"
```

#### 3. Autocomplete Prefix
```bash
curl "http://localhost:8000/autocomplete?prefix=ser"
```

#### 4. Prometheus Metrics Scrape
```bash
curl "http://localhost:8000/metrics"
```

---

## 7. Lightweight Host Deployment (systemd, No Docker)

If you find running the full Docker + Airflow + Prometheus + Grafana stack too heavyweight for simple local lookup needs, you can run the core system **directly on your host as a background systemd service** that starts automatically on system boot.

### Host Dependencies
Install Redis directly on your Linux host:
```bash
sudo apt update
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
```

### Step 1: Install Python Dependencies
Create a Python virtual environment inside the workspace and install requirements:
```bash
cd /home/kami/Desktop/codebase/rag-dict
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Create a systemd Service File
Create a new service configuration file at `/etc/systemd/system/rag-dict.service` (e.g., using `sudo nano`):
```ini
[Unit]
Description=LexiSeek RAG Dictionary Service
After=network.target redis-server.service

[Service]
Type=simple
User=kami
WorkingDirectory=/home/kami/Desktop/codebase/rag-dict
ExecStart=/home/kami/Desktop/codebase/rag-dict/venv/bin/uvicorn src.serve:app --host 127.0.0.1 --port 8000
Restart=always
Environment=PYTHONPATH=/home/kami/Desktop/codebase/rag-dict
Environment=REDIS_HOST=127.0.0.1
Environment=REDIS_PORT=6379
Environment=DICT_JSON=/home/kami/Desktop/codebase/rag-dict/data/dictionary.json
Environment=FAISS_PATH=/home/kami/Desktop/codebase/rag-dict/data/embeddings.faiss

[Install]
WantedBy=multi-user.target
```

### Step 3: Enable and Start the Background Service
Reload systemd, enable the service to start automatically on boot, and start it immediately:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-dict.service
sudo systemctl start rag-dict.service
```

### How the Host Deployment Operates
* **Self-Building**: On first startup, the FastAPI server detects if `dictionary.trie` or `embeddings.faiss` are missing and compiles them automatically (takes ~1-2 minutes).
* **Self-Warming**: FastAPI launches a background thread that silently pre-warms the host's local Redis cache in the background (takes ~3-5 seconds), keeping the API fully responsive instantly.
* **Super Lightweight**: Only Redis and the FastAPI process run in the background on the host (uses <300MB of RAM total vs. >3GB RAM for the full Docker/Airflow stack!).
* **Auto-Start**: systemd ensures the dictionary server starts immediately on system boot and restarts automatically if terminated.

