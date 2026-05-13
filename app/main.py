import logging

import pickle

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader 

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from prometheus_fastapi_instrumentator import Instrumentator

import uvicorn

from uuid import uuid4

from app.config import Config

from app.model_verification import verify_model

CATEGORY_MAPPING = {
    0: 'Atheism',
    1: 'Computer Graphics',
    2: 'Windows OS',
    3: 'IBM PC Hardware',
    4: 'Mac Hardware',
    5: 'Windows X Window System',
    6: 'Miscellaneous For Sale',
    7: 'Cars / Autos',
    8: 'Motorcycles',
    9: 'Baseball',
    10: 'Hockey',
    11: 'Cryptography',
    12: 'Electronics',
    13: 'Medicine',
    14: 'Space',
    15: 'Christianity',
    16: 'Gun Politics',
    17: 'Middle East Politics',
    18: 'General Politics',
    19: 'Miscellaneous Religion',
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Newsgroup Analysis API",
    description="API for classifying text messages into newsgroup categories using a MultinomialNB",
    version="1.0.0"
)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Adds a unique ID to each request"""
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    
    logger.info(f"[{request_id}] -> {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    response.headers["X-Request-ID"] = request_id
    
    logger.info(f"[{request_id}] <- {response.status_code}")
    
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",
        "http://127.0.0.1:7860"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.state.limiter = limiter 
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) #type: ignore

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str = Depends(api_key_header)
):
    """Checks the API key."""
    request_id = request.state.request_id

    if not api_key:
        logger.warning(f"[{request_id}] Request missing API key")
        raise HTTPException(
            status_code=403,
            detail="Missing API Key. Please provide X-API-Key header"
        )
    
    if not Config.is_valid_api_key(api_key):
        logger.warning(f"[{request_id}] Invalid API key attempt")
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    
    return api_key


def check_ip_allowed(request: Request):
    """Checks the client's IP address."""
    request_id = request.state.request_id
    client_ip = get_remote_address(request)
    
    if not Config.is_ip_allowed(client_ip):
        logger.warning(f"[{request_id}] Blocked request from unauthorized IP: {client_ip}")
        raise HTTPException(
            status_code=403,
            detail=f"IP {client_ip} not allowed. Only whitelisted IPs can access this API."
        )
    
    return client_ip


model = None
MODEL_PATH = Config.MODEL_PATH


def load_model(path_to_model: Path) -> None:
    """Load the pre-trained model from a pickle file."""
    global model
    try:
        with open(path_to_model, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model successfully loaded")
    except FileNotFoundError:
        logger.error(f"Model file not found at {path_to_model}")
    except Exception as e:
        logger.error(f"Error loading model {e}")

    
def check_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


if verify_model():
    load_model(MODEL_PATH)
else:
    raise RuntimeError("Model verification failed. API will not start.")


class TextPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2100)

    """Incoming request schema."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "text": "Space shuttle launched yesterday from Kennedy Space Center"
        }
    })


class ModelResponse(BaseModel):
    """API response schema."""
    text_class: int
    category_name: str
    text: str


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the API and model are running properly."""
    check_model()
    return {"status": "ok", "message": "API is running and model is loaded"}


@app.post("/predict", tags=["Predict"])
@limiter.limit(Config.RATE_LIMIT)
async def predict_class(
    request: Request,
    text_request: TextPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    request_id = request.state.request_id
    client_ip = check_ip_allowed(request)

    check_model()

    try:
        logger.info(f"[{request_id}] Authorized request from {client_ip}: {text_request.text[:50]}...")

        prediction = model.predict([text_request.text])[0] # type: ignore

        response = ModelResponse(
            text_class=int(prediction),
            text=text_request.text,
            category_name=CATEGORY_MAPPING[int(prediction)]
            )

        logger.info(f"[{request_id}] Result for {client_ip}: {prediction}")
        
        return response
    
    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    

@app.get("/rate-limit-info", tags=["Info"])
async def rate_limit_info():
    """Get information about API rate limits."""
    return {
        "endpoint": "/predict",
        "limit": "100 requests per minute",
        "per": "IP address",
        "status": "active"
    }


@app.get("/config-info", tags=["Info"])
async def config_info():
    """Get information about API configuration"""
    return Config.get_api_keys_info()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
