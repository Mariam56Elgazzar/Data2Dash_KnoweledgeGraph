# Data2Dash – Knowledge Graph Extractor (InsightGraph)

Data2Dash is a Streamlit app that turns research papers (PDF/TXT) or pasted text into an interactive **knowledge graph**.  
It extracts **entities + relationships** (models, datasets, metrics, methods, results, limitations, etc.), renders them in **PyVis**, and can optionally **sync to Neo4j**.

---

## ✨ Features

- 📄 Upload **PDF** or **TXT**, or paste text manually
- 🧠 Extract entities + relations using **Groq (Llama 3.1)** via LangChain
- 🧩 Multiple chunk strategies (sections + sliding windows + page markers)
- 🧹 Preprocessing for academic papers (cleanup, stop at references, normalization)
- 🌐 Interactive graph visualization in the browser (PyVis)
- 🔗 Optional: Sync extracted graph into **Neo4j**

---

## 🧱 Project Structure

Recommended structure:

```

data2dash/
├─ app.py
├─ insightgraph_pipeline.py
├─ preprocessing.py
├─ requirements.txt
├─ .env
└─ README.md

````

---

## ✅ Requirements

- Python **3.10+** (recommended: 3.10 or 3.11)
- A **Groq API key**
- (Optional) **Neo4j** running locally or remotely

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here

# Optional Neo4j (only if you enable syncing)
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
````

> If you don’t want Neo4j sync, you can omit the Neo4j variables.

---

## 📦 Installation

### 1) Create and activate a virtual environment

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the local URL that Streamlit prints (usually `http://localhost:8501`).

---

## ⚙️ Usage

1. Choose input method in the sidebar:

   * **Upload PDF/TXT**
   * **Manual text input**
2. Click **🚀 Generate Knowledge Graph**
3. Explore the graph:

   * drag nodes
   * zoom in/out
   * hover to view types and relations
4. Download the generated HTML graph from the button.

---

## 🔗 Neo4j Sync (Optional)

If you want to push the extracted graph into Neo4j:

1. Start Neo4j (Desktop or Docker)
2. Add Neo4j credentials in `.env` (or in the Streamlit sidebar)
3. Enable **“Sync to Neo4j Database”** in the app

### Docker Neo4j quick start

```bash
docker run -it --rm \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

Open Neo4j Browser at `http://localhost:7474`.

---

## 🧪 Recommended `requirements.txt`

If you don’t have one yet, here is a stable baseline:

```txt
streamlit>=1.32.0
python-dotenv>=1.0.1
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-experimental>=0.0.60
langchain-groq>=0.1.5
pydantic>=2.6.0
pyvis>=0.3.2
neo4j>=5.20.0
pypdf>=4.0.0
```

> If you hit version conflicts, pin versions based on your working environment.

---

## 🛠 Troubleshooting

### 1) “Rate limit reached” / Too many requests (Groq)

* Lower **Concurrency** in sidebar (e.g., 2–4)
* Lower **Max chunks**
* Use fewer overlapping chunk strategies

### 2) Empty graph / 0 relationships

* Try increasing **Max chunks**
* Upload a clearer PDF (some PDFs contain images only or broken text)
* Ensure preprocessing didn’t remove too much content
* If PDF extraction is weak, convert PDF → text using a stronger extractor (Grobid / PyMuPDF)

### 3) `ModuleNotFoundError: No module named 'pyvis'`

```bash
pip install pyvis
```

### 4) Neo4j sync fails

* Confirm Neo4j is running
* Confirm URL (`bolt://localhost:7687`)
* Confirm username/password
* Check firewall/ports if remote

### 5) Streamlit async errors (`asyncio.run() cannot be called...`)

* Use the included **production pipeline** (`insightgraph_pipeline.py`) which handles event loop cases
* If still failing, run extraction synchronously (we can patch it)

---

## 🔒 Notes on Privacy / Data

* Uploaded PDFs are saved temporarily for processing and then removed.
* Your Groq key stays local in your environment.
* Do not upload sensitive documents unless you understand where your API requests go.

---

## 📌 Roadmap (Optional Ideas)

* Better PDF extraction (PyMuPDF / Grobid)
* Entity merging using embeddings (reduce duplicates)
* Export formats: JSON-LD, GraphML
* Improve Neo4j schema + constraints
* Add search / filtering inside the graph UI

---

