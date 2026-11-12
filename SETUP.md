# GreenRouting RAG — Setup and Run Manual

This guide is for a new team member who has never run the project before. The normal setup is local: Python runs the application, Ollama runs the language models, and Streamlit opens the user interface in a browser. Redis is used for the semantic cache when available; SQLite remains the durable fallback.

## 1. What you need to install

Install these items once:

1. **Python 3.10 or 3.11**
2. **Ollama** (required for real local AI answers)
3. **Git** (optional; only needed if you download the project with Git)

Redis is optional for a direct Python run. If it is unavailable, the semantic cache continues to work from SQLite. Docker Compose starts Redis automatically with the app.

Minimum practical computer: 8 GB RAM and about 20 GB free disk space. A modern 4-core CPU works, but 16 GB RAM and an SSD make the demo much faster. A GPU is optional.

## 2. Open the project folder

Open PowerShell and run:

```powershell
Set-Location "D:\Innovative Project-Sem 7\GreenRoutingRag"
```

If the project is stored somewhere else, replace the path with that folder.

## 3. Create a Python virtual environment

The virtual environment keeps this project's packages separate from other Python projects.

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python --version
```

The last command should show Python 3.10 or 3.11. Run the activation command again whenever a new PowerShell window is opened.

If `py -3.11` is not available, use the Python launcher version installed on the computer, for example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. Install Python packages

With `(.venv)` visible at the beginning of the PowerShell prompt, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first installation can take several minutes because FAISS, PyTorch, and Sentence Transformers have larger dependencies.

## 5. Install and prepare Ollama

Install Ollama from the [official Ollama download page](https://ollama.com/download). After installation, open a new PowerShell window and check it:

```powershell
ollama --version
ollama list
```

The current configuration expects two local models:

```powershell
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
```

Check that they are present:

```powershell
ollama list
```

If Ollama Desktop is already running, its local service normally starts automatically. If it is not running, start the service in another PowerShell window:

```powershell
ollama serve
```

Do not run a second `ollama serve` if the service is already using port 11434.

The names above are configured in `config/config.yaml`:

```yaml
lightweight_model: "qwen2.5:0.5b"
capable_model: "llama3.2:3b"
```

If the team chooses different Ollama models, change those two values to the exact names shown by `ollama list`.

## 6. Start the application

To use Redis during a direct local Python run, start the Redis service first (for example, `docker compose up -d redis`). The default connection is `redis://localhost:6379/0`; set `REDIS_URL` to use a different Redis server. If the app cannot connect, its sidebar reports that SQLite fallback is active.

From the project folder, with the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Streamlit will print a local address. Open this address in a browser, normally:

```text
http://localhost:8501
```

The **Experimental Benchmarking** tab also includes Optuna threshold tuning. It runs several
GreenRouting-only benchmark trials against the questions entered on that page. Keep Ollama
running and load a document first; each trial sends all listed questions to the local model.
The tuner restores the current thresholds afterward and only displays a recommendation. It
uses the project's heuristic quality score, so treat the result as experimental and do not
present simulated model runs as live energy measurements.

To stop the application, return to PowerShell and press `Ctrl+C`.

## 7. Use the application

1. Open **Document Knowledge Base**.
2. Upload one or more **text-based PDF, TXT, or Markdown (`.md`) files**.
3. Click **Process and Index Uploaded Files**.
4. Open **Green RAG Assistant**.
5. Ask a question whose answer is present in the uploaded files.
6. Ask the same question using different words to demonstrate semantic caching.
7. Check the displayed model, cache status, latency, estimated energy, and estimated CO2e.
8. Use **Experimental Benchmarking** to compare standard RAG with GreenRouting.

On the first run, the embedding model `all-MiniLM-L6-v2` may be downloaded from Hugging Face. Internet is needed for this first download; later runs can use the local cache.

## 8. Run the tests

```powershell
python -m pytest -q
```

Run tests after changing routing, caching, document processing, or energy-measurement code.

## 9. If Ollama is unavailable

The code has a deterministic simulation fallback enabled in `config/config.yaml`. The interface can still open and demonstrate the flow, but the answer is **not produced by a real language model**. The sidebar will show that Ollama is offline.

For the final demonstration and benchmark, keep Ollama connected and show the real model names. Simulation results must not be presented as real model energy measurements.

## 10. Common problems

### PowerShell refuses to activate `.venv`

Run this in the same PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `ollama: command not found`

Install Ollama, open a new terminal, and run `ollama --version` again. On Windows, restarting the terminal after installation is important.

### The sidebar says Ollama is offline

Start Ollama Desktop or run `ollama serve` in another terminal. Then confirm the models with `ollama list`.

### A model cannot be found

Run the matching pull command again, or update `lightweight_model` and `capable_model` in `config/config.yaml` to names shown by `ollama list`.

### The application is slow

This is expected on CPU-only machines or when the project/data is on an HDD. Use the lightweight model, keep the demo documents small, and move the project plus model storage to an SSD when possible.

### Port 8501 is already in use

Start Streamlit on another port:

```powershell
python -m streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502`.

### A scanned PDF cannot be indexed

The current document processor extracts selectable text. Image-only/scanned PDFs need OCR, which is planned as a later enhancement.

## 11. Run with Docker

Docker runs the Streamlit application in a container. Ollama remains installed on the host computer, so the model files do not get copied into the Docker image.

1. Install and start Docker Desktop.
2. Install Ollama on the host and make sure both configured models are downloaded with the `ollama pull` commands above.
3. From the project folder, build and start the app. Docker Compose starts Redis and the Streamlit app:

   ```powershell
   docker compose up --build -d
   ```

4. Open <http://localhost:8501>. To view app logs, run:

   ```powershell
   docker compose logs -f app
   ```

The container connects to host Ollama at `host.docker.internal:11434`. If the app reports Ollama as offline, configure the host Ollama service to listen on `0.0.0.0:11434`, restart Ollama, and retry. Set `OLLAMA_BASE_URL` in `docker-compose.yml` if Ollama runs at a different address.

Documents, SQLite data, and the FAISS index are stored in the `greenrouting-data` Docker volume. The embedding model download is stored separately in `huggingface-cache`. `docker compose down` stops the app but keeps these volumes. `docker compose down -v` removes the volumes and deletes their contents.

## 12. Linux/macOS command equivalent

The same project can run on Linux or macOS with these commands:

```bash
cd /path/to/GreenRoutingRag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
python -m streamlit run app.py
```

## 13. What gets created locally

After the first run, the application creates local files under `data/`, including the SQLite database and FAISS index. These files are machine-specific runtime data and are ignored by Git. Do not commit them to the repository.

