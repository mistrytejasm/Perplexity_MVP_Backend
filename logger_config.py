import logging
import sys

def setup_logging():
    """Setup simple logging configuration"""
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)  # Console output
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("services.tavily_service").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Less noise from HTTP requests
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Less noise from access logs
    
    print("✅ Logging configured successfully")

def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(name)


logger = setup_logging()