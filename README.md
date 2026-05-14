# Newsgroup Classification API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135.1-green.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-6.13.0-orange.svg)](https://gradio.app/)

Simple text classification API using MultinomialNB for newsgroup categories.

## Quick Start
```bash
# Clone repository
git clone https://github.com/uwaspwned/newsgroup-website.git

cd newsgroup-website

python -m venv .venv

source .venv/bin/activate  # for Linux/Mac
# or
.venv\Scripts\activate     # for Windows

pip install -r requirements.txt

# create .env file
cp .env.example .env

# Create model's keys
python -m ml.generate_keys

# Train model
python -m ml.train

# run API
uvicorn app.main:app --reload

# run another terminal for Gradio interface
python -m app.app

# open the site at the link http://127.0.0.1:7860 
```

## Example requests
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: gradio-secret-key-12345" \
  -d '{"text": "Space travel"}'
```

## Monitoring

### Metrics endpoint (Prometheus format)
```bash
curl http://localhost:8000/metrics
```

## View metrics with Prometheus
```bash
docker-compose up -d
open http://localhost:9090
```


## TODO

### Done
- [x] Text classification with MultinomialNB
- [x] Basic FastAPI setup
- [x] Category mapping (20 newsgroups)
- [x] Rate limiting (100/min)
- [x] CORS middleware
- [x] Gradio web interface
- [x] API Key authentication
- [x] IP whitelist
- [x] Environment config (`.env`)
- [x] Prometheus metrics (`/metrics` endpoint)
- [x] Prometheus server and grafana support (docker-compose)
- [x] Request ID tracking
- [x] Model verification
- [x] Input validation (sanitization)

### In Progress
- [ ] Response caching

### Planned
- [ ] HTTPS support
- [ ] Docker deployment

---

**Status:** 🟢 Active development
