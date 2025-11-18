"""Logging configuration and custom handlers"""
import logging
import threading
from transformers import logging as hf_logging

# Set logging to INFO level for cleaner output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom logger handler to capture agentic thoughts
class ThoughtCaptureHandler(logging.Handler):
    """Custom handler to capture internal thoughts from MedSwin and supervisor"""
    def __init__(self):
        super().__init__()
        self.thoughts = []
        self.lock = threading.Lock()
    
    def emit(self, record):
        """Capture log messages that contain agentic thoughts"""
        try:
            msg = self.format(record)
            # Only capture messages from GEMINI SUPERVISOR or MEDSWIN
            if "[GEMINI SUPERVISOR]" in msg or "[MEDSWIN]" in msg or "[MAC]" in msg:
                # Remove timestamp and logger name for cleaner display
                parts = msg.split(" - ", 3)
                if len(parts) >= 4:
                    clean_msg = parts[-1]
                else:
                    clean_msg = msg
                with self.lock:
                    self.thoughts.append(clean_msg)
        except Exception:
            pass
    
    def get_thoughts(self):
        """Get all captured thoughts as a formatted string"""
        with self.lock:
            return "\n".join(self.thoughts)
    
    def clear(self):
        """Clear captured thoughts"""
        with self.lock:
            self.thoughts = []

# Set MCP client logging to WARNING to reduce noise
mcp_client_logger = logging.getLogger("mcp.client")
mcp_client_logger.setLevel(logging.WARNING)
hf_logging.set_verbosity_error()

