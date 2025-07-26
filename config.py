# Flask Configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # RAG System Configuration
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    CHAT_MODEL = os.environ.get('CHAT_MODEL', 'gpt-4')
    VECTOR_STORE_PATH = os.environ.get('VECTOR_STORE_PATH', './bengali_translated_english_db')
    
    # Translation Configuration
    TRANSLATION_RATE_LIMIT = float(os.environ.get('TRANSLATION_RATE_LIMIT', '0.2'))
    DEFAULT_TRANSLATION_CONFIDENCE = float(os.environ.get('DEFAULT_TRANSLATION_CONFIDENCE', '0.9'))
    
    # Retrieval Configuration
    RETRIEVAL_K = int(os.environ.get('RETRIEVAL_K', '5'))
    RETRIEVAL_SCORE_THRESHOLD = float(os.environ.get('RETRIEVAL_SCORE_THRESHOLD', '0.1'))
    
    # OCR Configuration (if needed for future expansion)
    TESSERACT_PATH = os.environ.get('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    @staticmethod
    def validate_config():
        """Validate that required configuration is present"""
        required_vars = ['OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
