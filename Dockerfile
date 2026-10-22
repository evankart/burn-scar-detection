FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download Prithvi weights at build time so cold starts don't block on a 300MB fetch.
RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download("ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "Prithvi_EO_V2_300M.pt")
hf_hub_download("ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "prithvi_mae.py")
EOF

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
