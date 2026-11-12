# Run GreenRouting RAG with Docker (Windows)

This guide starts GreenRouting RAG and Redis with Docker Compose. **Ollama runs on the Windows computer, outside Docker**, and the app container connects to it. The Ollama models are downloaded on the computer, not baked into the app image.

## 1. Install Ollama and download the models

Install Ollama from the [official Windows download page](https://ollama.com/download/windows), or run this in PowerShell:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Open a new PowerShell window and download the two models expected by this project:

```powershell
ollama --version
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
ollama list
```

Keep Ollama running. Its Windows app normally starts the local model service automatically. Test that it responds:

```powershell
(Invoke-RestMethod http://localhost:11434/api/tags).models.name
```

You should see `qwen2.5:0.5b` and `llama3.2:3b` in the output.

## 2. Let the Docker app reach Ollama

The Compose file points the app to `http://host.docker.internal:11434`. Ollama normally listens only on `127.0.0.1`; to let the app container reach it, set its Windows `OLLAMA_HOST` environment variable to `0.0.0.0:11434`.

Run this in PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
```

Then fully quit Ollama from its system-tray icon and start it again from the Windows Start menu. Confirm the local API still responds with the command above.

> **Network note:** `0.0.0.0` makes Ollama listen on network interfaces. Use this on a trusted development computer/network, and avoid exposing port `11434` to the public internet. Ollama documents `OLLAMA_HOST` as the setting for changing its bind address. See the [Ollama server configuration FAQ](https://docs.ollama.com/faq#how-can-i-expose-ollama-on-my-network).

## 3. Install and start Docker Desktop

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) and start it. Docker Compose is included with Docker Desktop. Wait until Docker Desktop says its engine is running, then check from PowerShell:

```powershell
docker --version
docker compose version
```

## 4. Build and start GreenRouting

In PowerShell, move to the folder where you cloned this repository. For example:

```powershell
cd "D:\Innovative Project-Sem 7\GreenRoutingRag"
```

Build the app image and start the app plus Redis in the background:

```powershell
docker compose up --build -d
```

The first build can take several minutes. When it finishes, open:

```text
http://localhost:8501
```

The app should show Ollama as connected and Redis as connected in System Health. Uploaded documents and app data are stored in Docker volumes; Redis data and the embedding-model cache also use persistent volumes.

## 5. Useful Docker commands

Run these from the repository folder:

```powershell
# See running services
docker compose ps

# Follow the app logs
docker compose logs -f app

# Follow Redis logs
docker compose logs -f redis

# Stop the app and Redis; keep saved volumes and documents
docker compose down

# Rebuild after source-code or dependency changes
docker compose up --build -d
```

To completely remove the Compose volumes, including documents, the app database, Redis cache, and downloaded embedding-model cache, use:

```powershell
docker compose down -v
```

**This deletes the saved project data in those Docker volumes.** Ollama model files are stored separately by Ollama and are not removed by this command.

## Troubleshooting

### App opens, but Ollama says disconnected

1. Confirm Ollama runs on Windows: `(Invoke-RestMethod http://localhost:11434/api/tags).models.name`.
2. Confirm both expected models appear in `ollama list`.
3. Confirm `OLLAMA_HOST` is set to `0.0.0.0:11434`, then quit and restart Ollama.
4. Check app logs: `docker compose logs -f app`.
5. If Windows Firewall blocks the connection, allow Ollama on the trusted private network used by Docker Desktop; do not open it to public networks.

### Port 8501 is already in use

Stop the other app using port 8501, or change the left side of the `app` port mapping in `docker-compose.yml` (for example, `127.0.0.1:8502:8501`) and open that new local port.

### The app starts in simulation/fallback mode

Check the Ollama connection steps above. Without a reachable Ollama server and the configured models, Docker can still start the web app, but it cannot provide the intended live local-model answers.
