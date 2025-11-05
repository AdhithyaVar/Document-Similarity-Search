"""
Configuration Settings
Centralized configuration for the plagiarism detection system
"""

import os
from pathlib import Path

# Application Metadata
APP_TITLE = "Document Similarity Search"
APP_ICON = "📄"
VERSION = "2.0.0"

# File Processing
SUPPORTED_FORMATS = ["pdf", "docx", "txt", "csv", "xlsx"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_PAGES = 500  # Maximum pages to process

# Model Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE_DIR = Path.home() / ".cache" / "sentence-transformers"
EMBEDDING_DIMENSION = 384

# Text Processing
CHUNK_SIZE = 500  # Default chunk size in tokens
CHUNK_OVERLAP = 50  # Overlap between chunks
MIN_CHUNK_LENGTH = 50  # Minimum characters per chunk
MAX_CHUNK_LENGTH = 2000  # Maximum characters per chunk

# Similarity Thresholds
SIMILARITY_THRESHOLDS = {
    "high": 0.80,  # High plagiarism risk
    "moderate": 0.50,  # Moderate similarity
    "low": 0.30,  # Low similarity
}

# Analysis Modes
ANALYSIS_MODES = {
    "Quick": {
        "chunk_size": 1000,
        "overlap": 25,
        "detail_level": "basic",
    },
    "Standard": {
        "chunk_size": 500,
        "overlap": 50,
        "detail_level": "standard",
    },
    "Deep": {
        "chunk_size": 250,
        "overlap": 100,
        "detail_level": "comprehensive",
    },
}

# Language Processing
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"
]
DEFAULT_LANGUAGE = "en"

# Stopwords for common languages
STOPWORDS_ENABLED = True
LEMMATIZATION_ENABLED = True

# Report Generation
REPORT_TITLE = "Plagiarism Analysis Report"
REPORT_AUTHOR = "AI Detection System"
REPORT_INCLUDE_METADATA = True
REPORT_INCLUDE_STATISTICS = True
REPORT_INCLUDE_HEATMAP = True

# Export Settings
EXPORT_PDF_DPI = 300
EXPORT_EXCEL_SHEET_NAME = "Analysis Results"
EXPORT_JSON_INDENT = 2

# Performance Settings
BATCH_SIZE = 32  # Batch size for embeddings
USE_GPU = True  # Use GPU if available
NUM_WORKERS = 4  # Parallel processing workers
CACHE_EMBEDDINGS = True  # Cache computed embeddings

# UI Configuration
UI_THEME = "light"
CHART_COLOR_SCHEME = "viridis"
SHOW_PROGRESS_BAR = True
SHOW_DEBUG_INFO = False

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "plagiarism_checker.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security
SANITIZE_INPUTS = True
VALIDATE_FILE_TYPES = True
SCAN_FOR_MALICIOUS_CONTENT = False

# Database (for future expansion)
ENABLE_DATABASE = False
DATABASE_URL = "sqlite:///plagiarism_history.db"

# API Configuration (for future expansion)
API_ENABLED = False
API_RATE_LIMIT = 100  # requests per hour
API_KEY_REQUIRED = False

# Advanced Features
ENABLE_CITATION_DETECTION = True
ENABLE_PARAPHRASE_DETECTION = True
ENABLE_TRANSLATION_DETECTION = False
ENABLE_BATCH_PROCESSING = True

# Error Handling
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
TIMEOUT = 300  # seconds for long operations

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMP_DIR = PROJECT_ROOT / "temp"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Create directories if they don't exist
for directory in [TEMP_DIR, EXPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Environment-specific overrides
ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    SHOW_DEBUG_INFO = False
    LOG_LEVEL = "WARNING"
    CACHE_EMBEDDINGS = True
    USE_GPU = True
elif ENV == "development":
    SHOW_DEBUG_INFO = True
    LOG_LEVEL = "DEBUG"
elif ENV == "testing":
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for testing
    BATCH_SIZE = 16

# Feature Flags
FEATURES = {
    "heatmap_visualization": True,
    "detailed_statistics": True,
    "matching_highlights": True,
    "export_pdf": True,
    "export_excel": True,
    "export_json": True,
    "analysis_history": True,
    "batch_comparison": False,  # Coming soon
    "api_access": False,  # Coming soon
}

# Constants
EPSILON = 1e-8  # Small value to prevent division by zero
MIN_TEXT_LENGTH = 100  # Minimum text length to analyze
MAX_TEXT_LENGTH = 10_000_000  # Maximum text length (10M chars)

# Validation Rules
VALIDATION_RULES = {
    "min_file_size": 100,  # bytes
    "max_file_size": MAX_FILE_SIZE,
    "allowed_extensions": SUPPORTED_FORMATS,
    "forbidden_patterns": [r"\.exe$", r"\.dll$", r"\.bat$"],
}

# User Messages
MESSAGES = {
    "file_too_large": f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)}MB",
    "file_too_small": "File is too small to analyze",
    "invalid_format": f"Invalid file format. Supported: {', '.join(SUPPORTED_FORMATS)}",
    "extraction_failed": "Failed to extract text from document",
    "analysis_failed": "Analysis failed. Please try again.",
    "analysis_success": "Analysis completed successfully!",
}

# Help Text
HELP_TEXT = {
    "similarity_threshold": "Minimum similarity score to flag as potential plagiarism",
    "chunk_size": "Size of text segments for analysis. Smaller = more detailed but slower",
    "analysis_mode": "Quick: Fast scan | Standard: Balanced | Deep: Comprehensive",
    "export_format": "Choose format for downloading analysis results",
}
