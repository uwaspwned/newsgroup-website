from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
import logging

from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_model():
    with open(Config.PUBLIC_KEY_PATH, "rb") as f:
        verify_key = VerifyKey(f.read())

    with open(Config.MODEL_PATH, "rb") as f:
        model = f.read()

    with open(f"{Config.MODEL_PATH}.sig", "rb") as f:
        signature = f.read()

    try:
        verify_key.verify(model, signature)
        logger.info("Model signature OK")
        return True
    except BadSignatureError:
        logger.error("ERROR: model was modified")
        return False