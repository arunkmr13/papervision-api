import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    PaperVision App Configuration Settings.
    Loads variables from .env file or environment variables.
    """
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API Key")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Primary Gemini model for extraction")
    GEMINI_FALLBACK_MODEL: str = Field(default="gemini-1.5-flash", description="Fallback Gemini model if primary model fails")
    
    # Heuristic Thresholds
    MIN_WIDTH: int = Field(default=50, description="Minimum width of extracted raster image in pixels")
    MIN_HEIGHT: int = Field(default=50, description="Minimum height of extracted raster image in pixels")
    MIN_AREA: int = Field(default=2500, description="Minimum area of extracted raster image in pixels")
    TOP_MARGIN_EXCLUSION_PCT: float = Field(default=10.0, description="Exclude top % of page height (header area)")
    BOTTOM_MARGIN_EXCLUSION_PCT: float = Field(default=10.0, description="Exclude bottom % of page height (footer area)")
    MAX_ASPECT_RATIO: float = Field(default=8.0, description="Exclude extremely narrow/wide images (aspect ratio threshold)")
    MAX_HASH_OCCURRENCES: int = Field(default=2, description="Exclude repeated image hashes appearing on more than N pages")
    
    # Storage Settings
    UPLOAD_DIR: str = Field(default="storage/uploads", description="Directory to store uploaded PDF files during processing")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

# Instantiate global settings object
settings = Settings()
