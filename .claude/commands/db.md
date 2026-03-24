Manage the Qdrant vector database Docker container.

**Start Qdrant:**
```bash
docker compose up -d
```

**Stop Qdrant:**
```bash
docker compose down
```

**Check status:**
```bash
docker compose ps
```

Qdrant UI is available at http://localhost:6333/dashboard after starting.
Data persists in `./qdrant_storage/` across restarts.
