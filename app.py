"""Main entry point for MedLLM Agent"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from logger import logger
from config import DEFAULT_MEDICAL_MODEL
import config
from models import initialize_medical_model, initialize_tts_model
from client import MCP_AVAILABLE
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
        logger.info("✅ Gemini MCP SDK is available")
        if config.GEMINI_API_KEY:
            logger.info(f"✅ GEMINI_API_KEY is set: {config.GEMINI_API_KEY[:10]}...{config.GEMINI_API_KEY[-4:]}")
            # Test MCP connection asynchronously (don't block startup)
            try:
                import asyncio
                from client import test_mcp_connection
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, schedule test in background
                        logger.info("ℹ️ Testing MCP connection in background...")
                    else:
                        # Test synchronously
                        result = loop.run_until_complete(test_mcp_connection())
                        if result:
                            logger.info("✅ MCP connection test passed - Gemini MCP is ready!")
                        else:
                            logger.warning("⚠️ MCP connection test failed - will use fallback methods")
                except Exception as e:
                    logger.warning(f"Could not test MCP connection: {e}")
            except Exception as e:
                logger.debug(f"MCP connection test skipped: {e}")
        else:
            logger.warning("⚠️ GEMINI_API_KEY not set - Gemini MCP features will not work")
            logger.warning("   Set it in Hugging Face Space secrets or environment variables")
    else:
        logger.info("ℹ️ Gemini MCP SDK not available - app will use fallback methods (direct API calls)")
        logger.info("   This is normal and the app will continue to work. MCP is optional.")
    
    logger.info("Model preloading complete!")
    demo = create_demo()
    demo.launch()
