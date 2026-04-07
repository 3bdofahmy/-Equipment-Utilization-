#  🦅 Eagle — Construction Equipment CV Pipeline

Real-time construction equipment detection, tracking, and utilization monitoring using **YOLOv11** + **FastAPI** + **Kafka** + **PostgreSQL**.

> Automate construction site intelligence with computer-vision accuracy and millisecond latency.

---

## 🎯 Features

- ✅ **Real-time Detection** — YOLOv11-powered equipment detection at ≥10 FPS on GPU
- ✅ **Multi-Object Tracking** — BoT-SORT/ByteTrack with persistent equipment IDs (EQ-XXXX format)
- ✅ **Motion Analysis** — Camera motion compensation + zone-based activity scoring
- ✅ **Activity Classification** — Digging, Swinging/Loading, Dumping, Traveling, Waiting
- ✅ **Utilization Analytics** — Real-time equipment idle/active tracking with PostgreSQL persistence
- ✅ **REST API + Dashboard** — FastAPI Swagger + interactive frontend
- ✅ **Event Streaming** — Kafka for asynchronous frame + detection pipelines
- ✅ **LLM Verification** — Optional confidence boost from OpenAI/Gemini/Groq/Ollama
- ✅ **Flexible Inference Backends** — PyTorch / ONNX / TensorRT

---

## � Demo Video

Watch the system in action:

🎬 **[Full Demo on Google Drive](https://drive.google.com/file/d/12g1mf3QjDEEXV5OWZVHJIAYBrmxvhqc7/view?usp=sharing)**

---

## �📁 Project Structure

```
.
├── core/
│   ├── enums.py          ← ALL string enums (EquipmentType, Activity, UtilizationState …)
│   ├── config.py         ← Pydantic settings — reads from .env automatically
│   └── logger.py         ← Shared logger
│
├── database/
│   ├── models.py         ← SQLAlchemy ORM models (Equipment, Detection, UtilizationSummary)
│   ├── connection.py     ← Async engine, session factory, detection batching
│   └── repository/
│       ├── equipment_repo.py
│       ├── detection_repo.py
│       └── utilization_repo.py
│
├── alembic/              ← Migration scripts (auto-generated)
│   ├── env.py            ← Wired to database/models.py Base
│   └── versions/         ← Migration files live here
├── alembic.ini
│
├── api/
│   ├── main.py           ← FastAPI app + lifespan
│   ├── schemas.py        ← Pydantic request/response models (use Enums)
│   ├── dependencies.py
│   └── routers/
│       ├── equipment.py
│       ├── detections.py
│       ├── utilization.py
│       ├── stream.py
│       ├── model.py
│       └── health.py
│
├── inference/            ← Model loading + inference backends
├── tracking/             ← Multi-object tracker
├── motion/
│   ├── analyzer.py       ← Per-track motion analysis
│   ├── activity.py       ← Rule-based activity classifier (uses Activity enum)
│   └── state.py          ← ACTIVE/INACTIVE state machine (uses UtilizationState enum)
│
├── pipeline/             ← Frame processing pipeline
├── streaming/            ← Kafka producer/consumer + schemas
├── services/             ← cv_service.py, analytics_service.py
├── frontend/             ← Static dashboard (HTML/CSS/JS)
│
├── Dockerfile            ← Single file, three targets: api | cv | analytics
├── docker-compose.yml    ← All services in one file
├── requirements.txt
└── .env.example          ← Copy to .env and fill in values
```

---

## 🚀 Quick Start (Docker — Recommended)

**Prerequisites:** Docker & Docker Compose installed

### 1️⃣ Clone & Configure

```bash
git clone <your-repo>
cd Eagle_task

cp .env.example .env
# Edit .env — at minimum set:
#   LLM_API_KEY, VIDEO_SOURCE, INFERENCE_DEVICE
nano .env
```️⃣ Add Model W

### 2. Add your model weights

```bash
mkdir -p weights
cp /path/to/yolov8s.pt weights/   
```

### 3️⃣ Start Everything

```bash
docker compose up --build
```

This will:
- Start PostgreSQL + Kafka
- Run `alembic upgrade head` (creates all tables automatically)
- Start CV service, Analytics service, and API

### 4️⃣ Open Dashboard & Docs

```
http://localhost:8000
```

API docs (Swagger):
```
http://localhost:8000/docs
```

---

## 🛠 Local Development (Without Docker)

### 1️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Start Infrastructure Only

```bash
docker compose up postgres kafka zookeeper -d
```

### 3️⃣ Configure Environment

```bash
cp .env.example .env
# Set DB_HOST=localhost, KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### 4️⃣ Run Database Migrations

```bash
alembic upgrade head
```

### 5️⃣ Start Services Individually

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — CV pipeline
python -u services/cv_service.py

# Terminal 3 — Analytics
python -u services/analytics_service.py
```

---

## 🗄 Database Migrations (Alembic)

**Auto-generate migrations whenever you modify `database/models.py`:**

```bash
# Auto-detect changes and create a migration file
alembic revision --autogenerate -m "describe what changed"

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# See migration history
alembic history

# See current DB state
alembic current
```

> **How it works:** `alembic/env.py` imports `from database.models import Base` which registers all your ORM models. Alembic compares `Base.metadata` against the live DB to find differences.

---

## 🔢 Core Enums Reference

All valid string values are **centralized** in `core/enums.py` — no magic strings! 

### Import & Use Enums:

```python
from core.enums import EquipmentType, UtilizationState, Activity

# Enum values (type-safe)
EquipmentType.EXCAVATOR                # "excavator"
UtilizationState.ACTIVE                # "ACTIVE"
Activity.DIGGING                       # "Digging"
Activity.SWINGING_LOADING              # "Swinging/Loading"

# Convert YOLO class_id → enum
eq_type = EquipmentType.from_class_id(0)   # → EquipmentType.EXCAVATOR

# Get full class map for ModelRegistry
class_map = EquipmentType.class_map()  # → {0: "excavator", 1: "excavator_arm", 2: "truck"}
```

---

## 🔢 Enums Reference

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| **GET** | `/health` | Service status | `{db: ok, kafka: ok, model: ok}` |
| **GET** | `/equipment` | All detected equipment | Array of equipment records |
| **GET** | `/equipment/{id}` | Single equipment by ID | Equipment object |
| **GET** | `/utilization` | All utilization summaries | Array with utilization % |
| **GET** | `/utilization/{id}` | Single equipment utilization | Utilization object with pct |
| **GET** | `/utilization/{id}/history?minutes=30` | Time-series data | List of historical records |
| **GET** | `/detections?minutes=10&limit=200` | Recent detections | Recent detection payloads |
| **GET** | `/stream/latest-frame` | Latest annotated frame | Base64 JPEG |
| **GET** | `/model/info` | Model metadata | Model name, version, classes |
| **GET** | `/model/performance` | Performance metrics | FPS, GPU memory, latency |
| **GET** | `/docs` | Interactive Swagger UI | OpenAPI documentation |

---

## ⚙️ Configuration (Environment Variables)

Set these in `.env` before running:

| Variable | Default | Options | Purpose |
|----------|---------|---------|---------|
| `INFERENCE_BACKEND` | `pytorch` | `pytorch`, `onnx`, `tensorrt` | Which backend to use for inference |
| `INFERENCE_DEVICE` | `cuda` | `cuda`, `cpu` | GPU or CPU |
| `TRACKER` | `botsort` | `botsort`, `bytetrack` | Tracking algorithm |
| `DETECTION_CONF` | `0.35` | 0.0–1.0 | Min detection confidence threshold |
| `LLM_PROVIDER` | `openai` | `openai`, `gemini`, `groq`, `ollama` | LLM for verification |
| `LLM_ENABLED` | `true` | `true`, `false` | Enable/disable LLM verification |
| `PROCESS_FPS` | `10` | Any positive int | Frames to process per second |
| `VIDEO_SOURCE` | `data/input.mp4` | File path or RTSP URL | Input video source |

---

## 🐛 Debugging & Troubleshooting

### Check if PostgreSQL is running:
```bash
docker compose exec postgres psql -U cvuser -d construction_cv -c "\dt"
```

### See applied migrations:
```bash
alembic current
alembic history
```

### View logs for a specific service:
```bash
docker compose logs -f cv_service
docker compose logs -f api_service
docker compose logs -f analytics_service
```

### Rebuild a single service:
```bash
docker compose up --build api_service
```

### Reset everything (warning: deletes data):
```bash
docker compose down -v
docker compose up --build
```

### Test API health:
```bash
curl http://localhost:8000/health
```

### Check Kafka connectivity:
```bash
docker compose exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

---

## 📦 Adding a New Equipment Type

### Step 1: Update Enum

Edit `core/enums.py`:

```python
class EquipmentType(str, Enum):
    EXCAVATOR     = "excavator"
    EXCAVATOR_ARM = "excavator_arm"
    TRUCK         = "truck"
    CRANE         = "crane"           # ← Add new type here
```

Also update `class_map()` method in the same class.

### Step 2: Generate Migration

Alembic auto-detects enum changes:

```bash
alembic revision --autogenerate -m "add crane equipment type"
alembic upgrade head
```

### Step 3: Retrain YOLO Model

- Collect & annotate crane images on Roboflow
- Train YOLOv11 with new class
- Update `MODEL_PATH` in `.env`

### Done! ✅

New equipment type is now live in the pipeline.

---

## 🌐 REST API Reference

  ------------ --------------------------- ----------------------------------
  **Method**   **Endpoint**                **Description**

  GET          /health                     DB + Kafka + model status

  GET          /equipment                  All detected equipment

  GET          /equipment/{id}             One equipment by ID

  GET          /utilization                All utilization summaries

  GET          /utilization/{id}           One summary

  GET          /utilization/{id}/history?minutes=30   Time-series data

  GET          /detections?minutes=10&limit=200       Recent detections

  GET          /stream/latest-frame        Latest annotated frame (base64 JPEG)

  GET          /model/info                 Model info

  GET          /model/performance          FPS, GPU usage

  GET          /docs                       Swagger UI
  ------------ --------------------------- ----------------------------------

---

## ⚙️ Configuration Reference

All settings come from `.env`. Key variables:

| Variable | Default | Options |
|----------|---------|---------|
| `INFERENCE_BACKEND` | `pytorch` | `pytorch`, `onnx`, `tensorrt` |
| `INFERENCE_DEVICE` | `cuda` | `cuda`, `cpu` |
| `TRACKER` | `botsort` | `botsort`, `bytetrack` |
| `LLM_PROVIDER` | `openai` | `openai`, `gemini`, `groq`, `ollama` |
| `LLM_ENABLED` | `true` | `true`, `false` |
| `PROCESS_FPS` | `10` | Any int |
| `VIDEO_SOURCE` | `data/input.mp4` | File path or RTSP URL |

---

## 🐛 Debugging Tips

**Check if DB is up:**
```bash
docker compose exec postgres psql -U cvuser -d construction_cv -c "\dt"
```

**Check migrations applied:**
```bash
alembic current
```

**View logs for one service:**
```bash
docker compose logs -f cv_service
docker compose logs -f api_service
```

**Rebuild only one service:**
```bash
docker compose up --build api_service
```

**Reset everything (delete volumes):**
```bash
docker compose down -v
docker compose up --build
```

**Test API health:**
```bash
curl http://localhost:8000/health
```

---

## 📦 Adding a New Equipment Type

1. Add to `core/enums.py`:
```python
class EquipmentType(str, Enum):
    EXCAVATOR     = "excavator"
    EXCAVATOR_ARM = "excavator_arm"
    TRUCK         = "truck"
    CRANE         = "crane"          # ← add here
```

2. Update `class_map()` in the same enum.

3. Generate a migration (Alembic auto-detects the enum change):
```bash
alembic revision --autogenerate -m "add crane equipment type"
alembic upgrade head
```

4. Retrain your YOLO model with the new class, update `MODEL_PATH`.