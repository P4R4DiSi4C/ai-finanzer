from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # These settings can be overridden by environment variables:
    # STOCK_API_HOST, STOCK_API_PORT, STOCK_API_DEBUG_MODE
    host: str = "0.0.0.0"
    port: int = 8000
    debug_mode: bool = True

    class Config:
        env_prefix = "STOCK_API_"  # Makes environment variables be prefixed with STOCK_API_
        env_file = ".env"  # Will also load settings from .env file if it exists

settings = Settings() 