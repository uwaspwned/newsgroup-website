import os
import ipaddress
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class Config:
    """App configuration"""
    
    GRADIO_API_KEY = os.getenv("GRADIO_API_KEY")
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

    VALID_API_KEYS = [
        GRADIO_API_KEY,
        ADMIN_API_KEY,
    ]
    
    VALID_API_KEYS = [key for key in VALID_API_KEYS if key is not None]

    ALLOWED_IPS = [
        "127.0.0.1",
        "localhost",
    ]

    EXTRA_ALLOWED_IPS = os.getenv("ALLOWED_IPS")
    if EXTRA_ALLOWED_IPS:
        ALLOWED_IPS.extend([
            ip.strip()
            for ip in EXTRA_ALLOWED_IPS.split(",")
            if ip.strip()
        ])
    
    SERVER_IP = os.getenv("SERVER_IP")
    if SERVER_IP:
        ALLOWED_IPS.append(SERVER_IP)

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() in ("1", "true", "yes")
    SSL_KEYFILE = os.getenv("SSL_KEYFILE")
    SSL_CERTFILE = os.getenv("SSL_CERTFILE")

    MODEL_PATH = Path(os.getenv("MODEL_PATH", "./models/model.pkl"))
    MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "./models/model_config.json"))

    PRIVATE_KEY_PATH = Path(os.getenv("PRIVATE_KEY_PATH", "./models/private.key"))
    PUBLIC_KEY_PATH = Path(os.getenv("PUBLIC_KEY_PATH", "./models/public.key"))

    RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")

    @classmethod
    def is_ip_allowed(cls, ip: str) -> bool:
        """Checks if an IP address is allowed"""
        if ip in cls.ALLOWED_IPS:
            return True
        
        if ip == "127.0.0.1" or ip == "::1":
            return True

        try:
            client_ip = ipaddress.ip_address(ip)
        except ValueError:
            return False

        for allowed_ip in cls.ALLOWED_IPS:
            try:
                if "/" in allowed_ip and client_ip in ipaddress.ip_network(allowed_ip, strict=False):
                    return True
            except ValueError:
                continue
        
        return False
    
    @classmethod
    def is_valid_api_key(cls, api_key: str) -> bool:
        """Checks if the API key is valid"""
        if not api_key:
            return False
        return api_key in cls.VALID_API_KEYS
    
    @classmethod
    def get_api_keys_info(cls) -> dict:
        """Returns information about keys"""
        return {
            "gradio_key_configured": bool(cls.GRADIO_API_KEY),
            "admin_key_configured": bool(cls.ADMIN_API_KEY),
            "total_keys": len(cls.VALID_API_KEYS),
            "allowed_ips_count": len(cls.ALLOWED_IPS),
        }


if not Config.VALID_API_KEYS:
    print("   WARNING: No API keys configured!")
    print("   Please create .env file with GRADIO_API_KEY")
    print("   Example: GRADIO_API_KEY=your-secret-key-here")
else:
    print(f"Config loaded: {len(Config.VALID_API_KEYS)} API keys, {len(Config.ALLOWED_IPS)} allowed IPs")
