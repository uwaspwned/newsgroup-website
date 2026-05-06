class Config:
    """App configuration"""
    
    VALID_API_KEYS = [
        "gradio-secret-key-12345",
        "admin-secret-key-67890",
    ]
    
    ALLOWED_IPS = [
        "127.0.0.1",
        "localhost",
        # Add your server's IP here
    ]
    
    @classmethod
    def is_ip_allowed(cls, ip: str) -> bool:
        """Checks if an IP address is allowed"""

        if ip in cls.ALLOWED_IPS:
            return True

        if ip == "127.0.0.1" or ip == "::1":
            return True
            
        return False
    
    @classmethod
    def is_valid_api_key(cls, api_key: str) -> bool:
        """Checks if the API key is valid"""
        return api_key in cls.VALID_API_KEYS