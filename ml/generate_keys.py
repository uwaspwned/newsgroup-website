from nacl.signing import SigningKey

from app.config import Config

PRIVATE_KEY_PATH = Config.PRIVATE_KEY_PATH
PUBLIC_KEY_PATH = Config.PUBLIC_KEY_PATH

PRIVATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
PUBLIC_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

signing_key = SigningKey.generate()
verify_key = signing_key.verify_key

with open(PRIVATE_KEY_PATH, "wb") as f:
    f.write(signing_key.encode())

with open(PUBLIC_KEY_PATH, "wb") as f:
    f.write(verify_key.encode())