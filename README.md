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
pip install -r requirements.txt

# run API
cd web_app
uvicorn main:app

# run Gradio interface (another terminal!)
cd web_app

python app.py

# open the site at the link http://127.0.0.1:7860 
```

## Example requests
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Space shuttle launched yesterday"}'
```

## TODO

### Done
- [x] Text classification with MultinomialNB
- [x] Basic FastAPI setup
- [x] Category mapping (20 newsgroups)
- [x] Rate limiting (100/min)
- [x] CORS middleware
- [x] Gradio web interface

### In Progress
- [ ] API Key authentication
- [ ] IP whitelist
- [ ] Environment config (`.env`)

### Planned
- [ ] Docker deployment
- [ ] Request ID tracking
- [ ] Input validation (sanitization)
- [ ] Model hash verification
- [ ] Response caching
- [ ] HTTPS support

---

**Status:** 🟢 Active development
