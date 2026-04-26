import logging
import pickle

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import uvicorn

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

model = None
MODEL_PATH = Path('model.pkl')


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


load_model(MODEL_PATH)


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
@limiter.limit("100/minute")
async def predict_class(request: Request, text_request: TextPredictionRequest):
    check_model()

    try:
        client_ip = get_remote_address(request)
        logger.info(f"Request from {client_ip}: {text_request.text[:50]}...")

        prediction = model.predict([text_request.text])[0] # type: ignore

        response = ModelResponse(
            text_class=int(prediction),
            text=text_request.text,
            category_name=CATEGORY_MAPPING[int(prediction)]
            )

        logger.info(f"Result for {client_ip}: {prediction}")
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)