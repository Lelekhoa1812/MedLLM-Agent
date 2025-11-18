"""Main entry point for MedLLM Agent"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from logger import logger
from config import DEFAULT_MEDICAL_MODEL
import config
from models import initialize_medical_model, initialize_tts_model
from mcp import MCP_AVAILABLE
from ui import create_demo

if __name__ == "__main__":
    # Preload models on startup
    logger.info("Preloading models on startup...")
    logger.info("Initializing default medical model (MedSwin TA)...")
    initialize_medical_model(DEFAULT_MEDICAL_MODEL)
    logger.info("Preloading TTS model...")
    try:
        initialize_tts_model()
        if config.global_tts_model is not None:
            logger.info("TTS model preloaded successfully!")
        else:
            logger.warning("TTS model not available - will use MCP or disable voice generation")
    except Exception as e:
        logger.warning(f"TTS model preloading failed: {e}")
        logger.warning("Text-to-speech will use MCP or be disabled")
    
    # Check Gemini MCP availability
    if MCP_AVAILABLE:
        logger.info("✅ Gemini MCP is available for translation, summarization, document parsing, and transcription")
    else:
        logger.info("ℹ️ Gemini MCP not available - app will use fallback methods (direct API calls)")
        logger.info("   This is normal and the app will continue to work. MCP is optional.")
    
    logger.info("Model preloading complete!")
    demo = create_demo()
    demo.launch()
