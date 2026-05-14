# Newsgroup Classification API

Text classification service for 20 Newsgroups categories. The project trains a
`TfidfVectorizer + MultinomialNB` pipeline, serves predictions through FastAPI,
provides a Gradio UI, verifies the model artifact signature, caches repeated
predictions in Redis, and exposes Prometheus metrics.

## Features

- Multinomial Naive Bayes text classifier.
- FastAPI prediction API with Pydantic validation.
- Gradio web interface.
- API key authentication.
- IP allowlist with CIDR support.
- Request ID tracking.
- Rate limiting.
- Redis response caching.
- Ed25519 model signature verification with PyNaCl.
- Prometheus metrics endpoint.
- Docker Compose setup for API, UI, Redis, Prometheus, and Grafana.
- Optional HTTPS support through Uvicorn SSL settings.

## Project Structure

```text
.
├── app/
│   ├── app.py                  # Gradio UI
│   ├── config.py               # Environment configuration
│   ├── main.py                 # FastAPI application
│   └── model_verification.py   # Model signature verification
├── ml/
│   ├── generate_keys.py        # Creates signing and verification keys
│   └── train.py                # Trains, evaluates, signs, and saves model
├── models/                     # Generated artifacts, ignored by Git
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── README.md
```

## Local Setup

```bash
git clone https://github.com/uwaspwned/newsgroup-website.git
cd newsgroup-website

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

Generate signing keys, train the model, and create model artifacts:

```bash
python -m ml.generate_keys
python -m ml.train
```

Start Redis cache:

```bash
docker compose up -d redis
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Start the Gradio UI in another terminal:

```bash
python -m app.app
```

Open:

- Gradio UI: http://localhost:7860
- API docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

## Docker Deployment

Create `.env` first:

```bash
cp .env.example .env
```

Then start the full stack:

```bash
docker compose up --build
```

Services:

- API: http://localhost:8000
- Gradio: http://localhost:7860
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Redis: localhost:6379

The API container bootstraps model artifacts when `models/` is empty:

1. Generates signing keys.
2. Trains the model.
3. Saves `model.pkl`.
4. Saves `model_config.json`.
5. Signs the model as `model.pkl.sig`.

The `models/` directory is mounted into the container so artifacts persist
between runs.

## Tests

Run the test suite:

```bash
pytest
```

The API tests use local generated artifacts from `models/`. If model artifacts
are missing, artifact-dependent tests are skipped.

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace_with_your_secret_key_here" \
  -d "{\"text\": \"Space shuttle launched yesterday from Kennedy Space Center\"}"
```

Example response:

```json
{
  "text_class": 14,
  "category_name": "Space",
  "text": "Space shuttle launched yesterday from Kennedy Space Center"
}
```

## Configuration

Main environment variables:

```env
GRADIO_API_KEY=replace_with_your_secret_key_here
ADMIN_API_KEY=replace_with_admin_secret_key_here

MODEL_PATH=./models/model.pkl
MODEL_CONFIG_PATH=./models/model_config.json
PRIVATE_KEY_PATH=./models/private.key
PUBLIC_KEY_PATH=./models/public.key

API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
FASTAPI_URL=http://localhost:8000
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860

ALLOWED_IPS=127.0.0.1,::1,172.16.0.0/12
RATE_LIMIT=100/minute

CACHE_ENABLED=true
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600
```

## HTTPS

For local HTTPS, provide a certificate and private key to Uvicorn.

Example with OpenSSL:

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout certs/key.pem \
  -out certs/cert.pem \
  -days 365 \
  -subj "/CN=localhost"
```

Set these variables in `.env`:

```env
SSL_KEYFILE=./certs/key.pem
SSL_CERTFILE=./certs/cert.pem
```

Then run:

```bash
python -m app.main
```

For real production deployments, TLS is usually terminated by a reverse proxy
or platform load balancer instead of directly inside the Python process.

## Model Verification

The model is signed after training with an Ed25519 private key. At API startup,
the public key verifies `model.pkl` against `model.pkl.sig`.

If verification fails, the API does not start.

Important production note: the private key should only be available during
model build or training. The inference service only needs the public key.

## Response Caching

Repeated predictions are cached in Redis. Cache keys are based on:

- sanitized input text;
- a namespace derived from the model signature.

This prevents stale cached predictions from being reused after the model is
retrained and re-signed.

Cache status:

```bash
curl http://localhost:8000/cache-info
```

If Redis is unavailable, the API continues working without cache.

## Model Metrics

Training writes model metrics to:

```text
models/model_config.json
```

Current local metrics include:

- accuracy;
- macro precision;
- macro recall;
- macro F1;
- full classification report.

The `models/` directory is ignored by Git because it contains generated model
artifacts and private signing keys.

## TODO

### Done

- [x] Text classification with MultinomialNB
- [x] Basic FastAPI setup
- [x] Category mapping for 20 Newsgroups
- [x] Rate limiting
- [x] CORS middleware
- [x] Gradio web interface
- [x] API key authentication
- [x] IP allowlist
- [x] Environment config
- [x] Prometheus metrics endpoint
- [x] Prometheus and Grafana support
- [x] Request ID tracking
- [x] Model verification
- [x] Input validation and sanitization
- [x] Redis response caching
- [x] HTTPS support
- [x] Docker deployment
- [x] API tests