import gradio as gr
import os
import base64
import logging
import torch
import threading
import time
import json
import concurrent.futures
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import logging as hf_logging
import spaces
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document as LlamaDocument,
)
from llama_index.core import Settings
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from langdetect import detect, LangDetectException
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
                # Format: "timestamp - logger - level - message"
                parts = msg.split(" - ", 3)
                if len(parts) >= 4:
                    clean_msg = parts[-1]  # Get the message part
                else:
                    clean_msg = msg
                with self.lock:
                    self.thoughts.append(clean_msg)
        except Exception:
            pass  # Ignore formatting errors
    
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

# MCP imports
MCP_CLIENT_INFO = None
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp import types as mcp_types
    from mcp.client.stdio import stdio_client
    import asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()  # Allow nested event loops
    except ImportError:
        pass  # nest_asyncio is optional
    MCP_AVAILABLE = True
    MCP_CLIENT_INFO = mcp_types.Implementation(
        name="MedLLM-Agent",
        version=os.environ.get("SPACE_VERSION", "local"),
    )
except ImportError as e:
    logger.warning(f"MCP SDK not available: {e}")
    MCP_AVAILABLE = False
    # Fallback imports if MCP is not available
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    TTS = None
import numpy as np
import soundfile as sf
import tempfile

# Model configurations
MEDSWIN_MODELS = {
    "MedSwin SFT": "MedSwin/MedSwin-7B-SFT",
    "MedSwin KD": "MedSwin/MedSwin-7B-KD",
    "MedSwin TA": "MedSwin/MedSwin-Merged-TA-SFT-0.7"
}
DEFAULT_MEDICAL_MODEL = "MedSwin TA"
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"  # Domain-tuned medical embedding model
TTS_MODEL = "maya-research/maya1"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

# Gemini MCP configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # Default for harder tasks
GEMINI_MODEL_LITE = os.environ.get("GEMINI_MODEL_LITE", "gemini-2.5-flash-lite")  # For parsing and simple tasks

# Custom UI
TITLE = "<h1><center>🩺 MedLLM Agent - Medical RAG & Web Search System</center></h1>"
DESCRIPTION = """
<center>
<p><strong>Advanced Medical AI Assistant</strong> powered by MedSwin models</p>
<p>📄 <strong>Document RAG:</strong> Answer based on uploaded medical documents</p>
<p>🌐 <strong>Web Search:</strong> Fetch knowledge from reliable online medical resources</p>
<p>🌍 <strong>Multi-language:</strong> Automatic translation for non-English queries</p>
<p>Upload PDF or text files to get started!</p>
</center>
"""
CSS = """
.upload-section {
    max-width: 400px;
    margin: 0 auto;
    padding: 10px;
    border: 2px dashed #ccc;
    border-radius: 10px;
}
.upload-button {
    background: #34c759 !important;
    color: white !important;
    border-radius: 25px !important;
}
.chatbot-container {
    margin-top: 20px;
}
.status-output {
    margin-top: 10px;
    font-size: 14px;
}
.processing-info {
    margin-top: 5px;
    font-size: 12px;
    color: #666;
}
.info-container {
    margin-top: 10px;
    padding: 10px;
    border-radius: 5px;
}
.file-list {
    margin-top: 0;
    max-height: 200px;
    overflow-y: auto;
    padding: 5px;
    border: 1px solid #eee;
    border-radius: 5px;
}
.stats-box {
    margin-top: 10px;
    padding: 10px;
    border-radius: 5px;
    font-size: 12px;
}
.submit-btn {
    background: #1a73e8 !important;
    color: white !important;
    border-radius: 25px !important;
    margin-left: 10px;
    padding: 5px 10px;
    font-size: 16px;
}
.input-row {
    display: flex;
    align-items: center;
}
.recording-timer {
    font-size: 12px;
    color: #666;
    text-align: center;
    margin-top: 5px;
}
.feature-badge {
    display: inline-block;
    padding: 3px 8px;
    margin: 2px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: bold;
}
.badge-rag {
    background: #e3f2fd;
    color: #1976d2;
}
.badge-web {
    background: #f3e5f5;
    color: #7b1fa2;
}
@media (min-width: 768px) {
    .main-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
    }
    .upload-section {
        flex: 1;
        max-width: 300px;
    }
    .chatbot-container {
        flex: 2;
        margin-top: 0;
    }
}
"""

# Global model storage
global_medical_models = {}
global_medical_tokenizers = {}
global_file_info = {}
global_tts_model = None
global_embed_model = None

# MCP client storage
global_mcp_session = None
global_mcp_stdio_ctx = None  # Store stdio context to keep it alive
global_mcp_lock = threading.Lock()  # Lock for thread-safe session access
# MCP server configuration via environment variables
# Gemini MCP server: Python-based server (agent.py)
# This works on Hugging Face Spaces without requiring npm/Node.js
# Make sure GEMINI_API_KEY is set in environment variables
# 
# Default configuration uses the bundled agent.py script
# To override:
#   export MCP_SERVER_COMMAND="python"
#   export MCP_SERVER_ARGS="/path/to/agent.py"
script_dir = os.path.dirname(os.path.abspath(__file__))
agent_path = os.path.join(script_dir, "agent.py")
MCP_SERVER_COMMAND = os.environ.get("MCP_SERVER_COMMAND", "python")
MCP_SERVER_ARGS = os.environ.get("MCP_SERVER_ARGS", agent_path).split() if os.environ.get("MCP_SERVER_ARGS") else [agent_path]

async def get_mcp_session():
    """Get or create MCP client session with proper context management"""
    global global_mcp_session, global_mcp_stdio_ctx
    
    if not MCP_AVAILABLE:
        logger.warning("MCP not available - SDK not installed")
        return None
    
    # Check if session exists and is still valid
    if global_mcp_session is not None:
        # Trust that existing session is valid - verify only when actually using it
        return global_mcp_session
    
    # Create new session using correct MCP SDK pattern
    try:
        # Prepare environment variables for MCP server
        mcp_env = os.environ.copy()
        if GEMINI_API_KEY:
            mcp_env["GEMINI_API_KEY"] = GEMINI_API_KEY
        else:
            logger.warning("GEMINI_API_KEY not set in environment. Gemini MCP features may not work.")
        
        # Add other Gemini MCP configuration if set
        if os.environ.get("GEMINI_MODEL"):
            mcp_env["GEMINI_MODEL"] = os.environ.get("GEMINI_MODEL")
        if os.environ.get("GEMINI_TIMEOUT"):
            mcp_env["GEMINI_TIMEOUT"] = os.environ.get("GEMINI_TIMEOUT")
        if os.environ.get("GEMINI_MAX_OUTPUT_TOKENS"):
            mcp_env["GEMINI_MAX_OUTPUT_TOKENS"] = os.environ.get("GEMINI_MAX_OUTPUT_TOKENS")
        if os.environ.get("GEMINI_TEMPERATURE"):
            mcp_env["GEMINI_TEMPERATURE"] = os.environ.get("GEMINI_TEMPERATURE")
        
        logger.info("Creating MCP client session...")
        
        server_params = StdioServerParameters(
            command=MCP_SERVER_COMMAND,
            args=MCP_SERVER_ARGS,
            env=mcp_env
        )
        
        # Correct MCP SDK usage: stdio_client is an async context manager
        # that yields (read, write) streams
        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        
        # Create ClientSession from the streams
        session = ClientSession(
            read,
            write,
            client_info=MCP_CLIENT_INFO,
        )
        
        # Initialize the session (this sends initialize request and waits for response + initialized notification)
        # The __aenter__() method handles the complete initialization handshake:
        # 1. Sends initialize request with client info
        # 2. Waits for initialize response from server
        # 3. Waits for initialized notification from server (this is critical!)
        # According to MCP protocol spec, the client MUST wait for the initialized notification
        # before sending any other requests (like list_tools)
        try:
            # The __aenter__() method properly handles the full initialization sequence
            # including waiting for the server's initialized notification
            # This is a blocking call that completes only after the server sends initialized
            await session.__aenter__()
            init_result = await session.initialize()
            server_info = getattr(init_result, "serverInfo", None)
            server_name = getattr(server_info, "name", "unknown")
            server_version = getattr(server_info, "version", "unknown")
            logger.info(f"✅ MCP session initialized (server={server_name} v{server_version})")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"❌ MCP session initialization failed: {error_type}: {error_msg}")
            
            # Clean up and return None
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            return None
        
        # Store both the session and stdio context to keep them alive
        global_mcp_session = session
        global_mcp_stdio_ctx = stdio_ctx
        logger.info("✅ MCP client session created successfully")
        return session
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"❌ Failed to create MCP client session: {error_type}: {error_msg}")
        global_mcp_session = None
        global_mcp_stdio_ctx = None
        return None

MCP_TOOLS_CACHE_TTL = int(os.environ.get("MCP_TOOLS_CACHE_TTL", "60"))
global_mcp_tools_cache = {"timestamp": 0.0, "tools": None}


def invalidate_mcp_tools_cache():
    """Invalidate cached MCP tool metadata"""
    global global_mcp_tools_cache
    global_mcp_tools_cache = {"timestamp": 0.0, "tools": None}


async def get_cached_mcp_tools(force_refresh: bool = False):
    """Return cached MCP tools list to avoid repeated list_tools calls"""
    global global_mcp_tools_cache
    if not MCP_AVAILABLE:
        return []
    
    now = time.time()
    if (
        not force_refresh
        and global_mcp_tools_cache["tools"]
        and now - global_mcp_tools_cache["timestamp"] < MCP_TOOLS_CACHE_TTL
    ):
        return global_mcp_tools_cache["tools"]
    
    session = await get_mcp_session()
    if session is None:
        return []
    
    try:
        tools_resp = await session.list_tools()
        tools_list = list(getattr(tools_resp, "tools", []) or [])
        global_mcp_tools_cache = {"timestamp": now, "tools": tools_list}
        return tools_list
    except Exception as e:
        logger.error(f"Failed to refresh MCP tools: {e}")
        invalidate_mcp_tools_cache()
        return []


async def call_agent(user_prompt: str, system_prompt: str = None, files: list = None, model: str = None, temperature: float = 0.2) -> str:
    """Call Gemini MCP generate_content tool"""
    if not MCP_AVAILABLE:
        logger.warning("MCP not available for Gemini call")
        return ""
    
    try:
        session = await get_mcp_session()
        if session is None:
            logger.warning("Failed to get MCP session for Gemini call")
            return ""
        
        tools = await get_cached_mcp_tools()
        if not tools:
            tools = await get_cached_mcp_tools(force_refresh=True)
        if not tools:
            logger.error("Unable to obtain MCP tool catalog for Gemini calls")
            return ""
        
        generate_tool = None
        for tool in tools:
            if tool.name == "generate_content" or "generate_content" in tool.name.lower():
                generate_tool = tool
                logger.info(f"Found Gemini MCP tool: {tool.name}")
                break
        
        if not generate_tool:
            logger.warning(f"Gemini MCP generate_content tool not found. Available tools: {[t.name for t in tools]}")
            invalidate_mcp_tools_cache()
            return ""
        
        # Prepare arguments
        arguments = {
            "user_prompt": user_prompt
        }
        if system_prompt:
            arguments["system_prompt"] = system_prompt
        if files:
            arguments["files"] = files
        if model:
            arguments["model"] = model
        if temperature is not None:
            arguments["temperature"] = temperature
        
        result = await session.call_tool(generate_tool.name, arguments=arguments)
        
        # Parse result
        if hasattr(result, 'content') and result.content:
            for item in result.content:
                if hasattr(item, 'text'):
                    response_text = item.text.strip()
                    return response_text
        logger.warning("⚠️ Gemini MCP returned empty or invalid result")
        return ""
    except Exception as e:
        logger.error(f"Gemini MCP call error: {e}")
        return ""

def initialize_medical_model(model_name: str):
    """Initialize medical model (MedSwin) - download on demand"""
    global global_medical_models, global_medical_tokenizers
    if model_name not in global_medical_models or global_medical_models[model_name] is None:
        logger.info(f"Initializing medical model: {model_name}...")
        model_path = MEDSWIN_MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        global_medical_models[model_name] = model
        global_medical_tokenizers[model_name] = tokenizer
        logger.info(f"Medical model {model_name} initialized successfully")
    return global_medical_models[model_name], global_medical_tokenizers[model_name]


def initialize_tts_model():
    """Initialize TTS model for text-to-speech"""
    global global_tts_model
    if not TTS_AVAILABLE:
        logger.warning("TTS library not installed. TTS features will be disabled.")
        return None
    if global_tts_model is None:
        try:
            logger.info("Initializing TTS model for voice generation...")
            global_tts_model = TTS(model_name=TTS_MODEL, progress_bar=False)
            logger.info("TTS model initialized successfully")
        except Exception as e:
            logger.warning(f"TTS model initialization failed: {e}")
            logger.warning("TTS features will be disabled. If pyworld dependency is missing, try: pip install TTS --no-deps && pip install coqui-tts")
            global_tts_model = None
    return global_tts_model


def get_or_create_embed_model():
    """Reuse embedding model to avoid reloading weights each request"""
    global global_embed_model
    if global_embed_model is None:
        logger.info("Initializing shared embedding model for RAG retrieval...")
        global_embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)
    return global_embed_model

async def transcribe_audio_gemini(audio_path: str) -> str:
    """Transcribe audio using Gemini MCP"""
    if not MCP_AVAILABLE:
        return ""
    
    try:
        # Ensure we have an absolute path
        audio_path_abs = os.path.abspath(audio_path)
        
        # Prepare file object for Gemini MCP using path (as per Gemini MCP documentation)
        files = [{
            "path": audio_path_abs
        }]
        
        # Use exact prompts from Gemini MCP documentation
        system_prompt = "You are a professional transcription service. Provide accurate, well-formatted transcripts."
        user_prompt = "Please transcribe this audio file. Include speaker identification if multiple speakers are present, and format it with proper punctuation and paragraphs, remove mumble, ignore non-verbal noises."
        
        result = await call_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            files=files,
            model=GEMINI_MODEL_LITE,  # Use lite model for transcription
            temperature=0.2
        )
        
        return result.strip()
    except Exception as e:
        logger.error(f"Gemini transcription error: {e}")
        return ""

def transcribe_audio(audio):
    """Transcribe audio to text using Gemini MCP"""
    if audio is None:
        return ""
    
    try:
        # Handle file path (Gradio Audio component returns file path)
        if isinstance(audio, str):
            audio_path = audio
        elif isinstance(audio, tuple):
            # Handle tuple format (sample_rate, audio_data)
            sample_rate, audio_data = audio
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, audio_data, samplerate=sample_rate)
                audio_path = tmp_file.name
        else:
            audio_path = audio
        
        # Use Gemini MCP for transcription
        if MCP_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    try:
                        import nest_asyncio
                        transcribed = nest_asyncio.run(transcribe_audio_gemini(audio_path))
                        if transcribed:
                            logger.info(f"Transcribed via Gemini MCP: {transcribed[:50]}...")
                            return transcribed
                    except Exception as e:
                        logger.error(f"Error in nested async transcription: {e}")
                else:
                    transcribed = loop.run_until_complete(transcribe_audio_gemini(audio_path))
                    if transcribed:
                        logger.info(f"Transcribed via Gemini MCP: {transcribed[:50]}...")
                        return transcribed
            except Exception as e:
                logger.error(f"Gemini MCP transcription error: {e}")
        
        logger.warning("Gemini MCP transcription not available")
        return ""
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

async def generate_speech_mcp(text: str) -> str:
    """Generate speech using MCP TTS tool"""
    if not MCP_AVAILABLE:
        return None
    
    try:
        # Get MCP session
        session = await get_mcp_session()
        if session is None:
            return None
        
        # Find TTS tool
        tools = await session.list_tools()
        tts_tool = None
        for tool in tools.tools:
            if "tts" in tool.name.lower() or "speech" in tool.name.lower() or "synthesize" in tool.name.lower():
                tts_tool = tool
                logger.info(f"Found MCP TTS tool: {tool.name}")
                break
        
        if tts_tool:
            result = await session.call_tool(
                tts_tool.name,
                arguments={"text": text, "language": "en"}
            )
            
            # Parse result - MCP might return audio data or file path
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        # If it's a file path
                        if os.path.exists(item.text):
                            return item.text
                    elif hasattr(item, 'data') and item.data:
                        # If it's binary audio data, save it
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(item.data)
                            return tmp_file.name
        return None
    except Exception as e:
        logger.warning(f"MCP TTS error: {e}")
        return None

def generate_speech(text: str):
    """Generate speech from text using TTS model (with MCP fallback)"""
    if not text or len(text.strip()) == 0:
        return None
    
    # Try MCP first if available
    if MCP_AVAILABLE:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                try:
                    import nest_asyncio
                    audio_path = nest_asyncio.run(generate_speech_mcp(text))
                    if audio_path:
                        logger.info("Generated speech via MCP")
                        return audio_path
                except:
                    pass
            else:
                audio_path = loop.run_until_complete(generate_speech_mcp(text))
                if audio_path:
                    return audio_path
        except Exception as e:
            pass  # MCP TTS not available, fallback to local
    
    # Fallback to local TTS model
    if not TTS_AVAILABLE:
        logger.error("TTS library not installed. Please install TTS to use voice generation.")
        return None
    
    global global_tts_model
    if global_tts_model is None:
        initialize_tts_model()
    
    if global_tts_model is None:
        logger.error("TTS model not available. Please check dependencies.")
        return None
    
    try:
        # Generate audio
        wav = global_tts_model.tts(text)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, wav, samplerate=22050)
            return tmp_file.name
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

def format_prompt_manually(messages: list, tokenizer) -> str:
    """Manually format prompt for models without chat template"""
    prompt_parts = []
    
    # Combine system and user messages into a single instruction
    system_content = ""
    user_content = ""
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content
        elif role == "assistant":
            # Skip assistant messages in history for now (can be added if needed)
            pass
    
    # Format for MedAlpaca/LLaMA-based medical models
    # Common format: Instruction + Input -> Response
    if system_content:
        prompt = f"{system_content}\n\nQuestion: {user_content}\n\nAnswer:"
    else:
        prompt = f"Question: {user_content}\n\nAnswer:"
    
    return prompt

def detect_language(text: str) -> str:
    """Detect language of input text"""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "en"  # Default to English if detection fails

def format_url_as_domain(url: str) -> str:
    """Format URL as simple domain name (e.g., www.mayoclinic.org)"""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        # Remove www. prefix if present, but keep it for display
        if domain.startswith('www.'):
            return domain
        elif domain:
            return domain
        return url
    except Exception:
        # Fallback: try to extract domain manually
        if '://' in url:
            domain = url.split('://')[1].split('/')[0]
            return domain
        return url

async def translate_text_gemini(text: str, target_lang: str = "en", source_lang: str = None) -> str:
    """Translate text using Gemini MCP"""
    if source_lang:
        user_prompt = f"Translate the following {source_lang} text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
    else:
        user_prompt = f"Translate the following text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
    
    # Use concise system prompt
    system_prompt = "You are a professional translator. Translate accurately and concisely."
    
    result = await call_agent(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,  # Use lite model for translation
        temperature=0.2
    )
    
    return result.strip()

def translate_text(text: str, target_lang: str = "en", source_lang: str = None) -> str:
    """Translate text using Gemini MCP"""
    if not MCP_AVAILABLE:
        logger.warning("Gemini MCP not available for translation")
        return text  # Return original text if translation fails
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                translated = nest_asyncio.run(translate_text_gemini(text, target_lang, source_lang))
                if translated:
                    logger.info(f"Translated via Gemini MCP: {translated[:50]}...")
                    return translated
            except Exception as e:
                logger.error(f"Error in nested async translation: {e}")
        else:
            translated = loop.run_until_complete(translate_text_gemini(text, target_lang, source_lang))
            if translated:
                logger.info(f"Translated via Gemini MCP: {translated[:50]}...")
                return translated
    except Exception as e:
        logger.error(f"Gemini MCP translation error: {e}")
    
    # Return original text if translation fails
    return text

async def search_web_mcp_tool(query: str, max_results: int = 5) -> list:
    """Search web using MCP web search tool (e.g., DuckDuckGo MCP server)"""
    if not MCP_AVAILABLE:
        return []
    
    try:
        tools = await get_cached_mcp_tools()
        if not tools:
            return []
        
        search_tool = None
        for tool in tools:
            tool_name_lower = tool.name.lower()
            if any(keyword in tool_name_lower for keyword in ["search", "duckduckgo", "ddg", "web"]):
                search_tool = tool
                logger.info(f"Found web search MCP tool: {tool.name}")
                break
        
        if not search_tool:
            tools = await get_cached_mcp_tools(force_refresh=True)
            for tool in tools:
                tool_name_lower = tool.name.lower()
                if any(keyword in tool_name_lower for keyword in ["search", "duckduckgo", "ddg", "web"]):
                    search_tool = tool
                    logger.info(f"Found web search MCP tool after refresh: {tool.name}")
                    break
        
        if search_tool:
            try:
                session = await get_mcp_session()
                if session is None:
                    return []
                # Call the search tool
                result = await session.call_tool(
                    search_tool.name,
                    arguments={"query": query, "max_results": max_results}
                )
            
                # Parse result
                web_content = []
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            try:
                                data = json.loads(item.text)
                                if isinstance(data, list):
                                    for entry in data[:max_results]:
                                        web_content.append({
                                            'title': entry.get('title', ''),
                                            'url': entry.get('url', entry.get('href', '')),
                                            'content': entry.get('body', entry.get('snippet', entry.get('content', '')))
                                        })
                                elif isinstance(data, dict):
                                    if 'results' in data:
                                        for entry in data['results'][:max_results]:
                                            web_content.append({
                                                'title': entry.get('title', ''),
                                                'url': entry.get('url', entry.get('href', '')),
                                                'content': entry.get('body', entry.get('snippet', entry.get('content', '')))
                                            })
                                    else:
                                        web_content.append({
                                            'title': data.get('title', ''),
                                            'url': data.get('url', data.get('href', '')),
                                            'content': data.get('body', data.get('snippet', data.get('content', '')))
                                        })
                            except json.JSONDecodeError:
                                # If not JSON, treat as plain text
                                web_content.append({
                                    'title': '',
                                    'url': '',
                                    'content': item.text[:1000]
                                })
                
                if web_content:
                    return web_content
            except Exception as e:
                logger.error(f"Error calling web search MCP tool: {e}")
        
        else:
            logger.debug("No MCP web search tool discovered in current catalog")
            return []
    except Exception as e:
        logger.error(f"Web search MCP tool error: {e}")
        return []

async def search_web_mcp(query: str, max_results: int = 5) -> list:
    """Search web using MCP tools - tries web search MCP tool first, then falls back to direct search"""
    # First try to use a dedicated web search MCP tool (like DuckDuckGo MCP server)
    results = await search_web_mcp_tool(query, max_results)
    if results:
        logger.info(f"✅ Web search via MCP tool: found {len(results)} results")
        return results
    
    # If no web search MCP tool available, use direct search (ddgs)
    # Note: Gemini MCP doesn't have web search capability, so we use direct API
    # The results will then be summarized using Gemini MCP
    logger.info("ℹ️ [Direct API] No web search MCP tool found, using direct DuckDuckGo search (results will be summarized with Gemini MCP)")
    return search_web_fallback(query, max_results)

def search_web_fallback(query: str, max_results: int = 5) -> list:
    """Fallback web search using DuckDuckGo directly (when MCP is not available)"""
    logger.info(f"🔍 [Direct API] Performing web search using DuckDuckGo API for: {query[:100]}...")
    # Always import here to ensure availability
    try:
        from ddgs import DDGS
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("Fallback dependencies (ddgs, requests, beautifulsoup4) not available")
        return []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            web_content = []
            for result in results:
                try:
                    url = result.get('href', '')
                    title = result.get('title', '')
                    snippet = result.get('body', '')
                    
                    # Try to fetch full content
                    try:
                        response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            # Extract main content
                            for script in soup(["script", "style"]):
                                script.decompose()
                            text = soup.get_text()
                            # Clean and limit text
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)
                            if len(text) > 1000:
                                text = text[:1000] + "..."
                            web_content.append({
                                'title': title,
                                'url': url,
                                'content': snippet + "\n" + text[:500] if text else snippet
                            })
                        else:
                            web_content.append({
                                'title': title,
                                'url': url,
                                'content': snippet
                            })
                    except:
                        web_content.append({
                            'title': title,
                            'url': url,
                            'content': snippet
                        })
                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue
            logger.info(f"✅ [Direct API] Web search completed: {len(web_content)} results")
            return web_content
    except Exception as e:
        logger.error(f"❌ [Direct API] Web search error: {e}")
        return []

def search_web(query: str, max_results: int = 5) -> list:
    """Search web using MCP tools (synchronous wrapper) - prioritizes MCP over direct ddgs"""
    # Always try MCP first if available
    if MCP_AVAILABLE:
        try:
            # Run async MCP search
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If loop is already running, use nest_asyncio or create new thread
                try:
                    import nest_asyncio
                    results = nest_asyncio.run(search_web_mcp(query, max_results))
                    if results:  # Only return if we got results from MCP
                        return results
                except (ImportError, AttributeError):
                    # Fallback: run in thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, search_web_mcp(query, max_results))
                        results = future.result(timeout=30)
                        if results:  # Only return if we got results from MCP
                            return results
            else:
                results = loop.run_until_complete(search_web_mcp(query, max_results))
                if results:  # Only return if we got results from MCP
                    return results
        except Exception as e:
            logger.error(f"Error running async MCP search: {e}")
    
    # Only use ddgs fallback if MCP is not available or returned no results
    logger.info("ℹ️ [Direct API] Falling back to direct DuckDuckGo search (MCP unavailable or returned no results)")
    return search_web_fallback(query, max_results)

async def summarize_web_content_gemini(content_list: list, query: str) -> str:
    """Summarize web search results using Gemini MCP"""
    combined_content = "\n\n".join([f"Source: {item['title']}\n{item['content']}" for item in content_list[:3]])
    
    user_prompt = f"""Summarize the following web search results related to the query: "{query}"
Extract key medical information, facts, and insights. Be concise and focus on reliable information.
Search Results:
{combined_content}
Summary:"""
    
    # Use concise system prompt
    system_prompt = "You are a medical information summarizer. Extract and summarize key medical facts accurately."
    
    result = await call_agent(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,  # Use full model for summarization
        temperature=0.5
    )
    
    return result.strip()

def summarize_web_content(content_list: list, query: str) -> str:
    """Summarize web search results using Gemini MCP"""
    if not MCP_AVAILABLE:
        logger.warning("Gemini MCP not available for summarization")
        # Fallback: return first result's content
        if content_list:
            return content_list[0].get('content', '')[:500]
        return ""
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                summary = nest_asyncio.run(summarize_web_content_gemini(content_list, query))
                if summary:
                    return summary
            except Exception as e:
                logger.error(f"Error in nested async summarization: {e}")
        else:
            summary = loop.run_until_complete(summarize_web_content_gemini(content_list, query))
            if summary:
                return summary
    except Exception as e:
        logger.error(f"Gemini MCP summarization error: {e}")
    
    # Fallback: return first result's content
    if content_list:
        return content_list[0].get('content', '')[:500]
    return ""

def get_llm_for_rag(temperature=0.7, max_new_tokens=256, top_p=0.95, top_k=50):
    """Get LLM for RAG indexing (uses medical model)"""
    # Use medical model for RAG indexing instead of translation model
    medical_model_obj, medical_tokenizer = initialize_medical_model(DEFAULT_MEDICAL_MODEL)
    
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=max_new_tokens,
        tokenizer=medical_tokenizer,
        model=medical_model_obj,
        generate_kwargs={
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
    )

async def autonomous_reasoning_gemini(query: str) -> dict:
    """Autonomous reasoning using Gemini MCP"""
    reasoning_prompt = f"""Analyze this medical query and provide structured reasoning:
Query: "{query}"
Analyze:
1. Query Type: (diagnosis, treatment, drug_info, symptom_analysis, research, general_info)
2. Complexity: (simple, moderate, complex, multi_faceted)
3. Information Needs: What specific information is required?
4. Requires RAG: (yes/no) - Does this need document context?
5. Requires Web Search: (yes/no) - Does this need current/updated information?
6. Sub-questions: Break down into key sub-questions if complex
Respond in JSON format:
{{
    "query_type": "...",
    "complexity": "...",
    "information_needs": ["..."],
    "requires_rag": true/false,
    "requires_web_search": true/false,
    "sub_questions": ["..."]
}}"""
    
    # Use concise system prompt
    system_prompt = "You are a medical reasoning system. Analyze queries systematically and provide structured JSON responses."
    
    response = await call_agent(
        user_prompt=reasoning_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,  # Use full model for reasoning
        temperature=0.3
    )
    
    # Parse JSON response (with fallback)
    try:
        # Extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            reasoning = json.loads(response[json_start:json_end])
        else:
            raise ValueError("No JSON found")
    except:
        # Fallback reasoning
        reasoning = {
            "query_type": "general_info",
            "complexity": "moderate",
            "information_needs": ["medical information"],
            "requires_rag": True,
            "requires_web_search": False,
            "sub_questions": [query]
        }
    
    logger.info(f"Reasoning analysis: {reasoning}")
    return reasoning

def autonomous_reasoning(query: str, history: list) -> dict:
    """
    Autonomous reasoning: Analyze query complexity, intent, and information needs.
    Returns reasoning analysis with query type, complexity, and required information sources.
    Uses Gemini MCP for reasoning.
    """
    if not MCP_AVAILABLE:
        logger.warning("⚠️ Gemini MCP not available for reasoning, using fallback")
        # Fallback reasoning
        return {
            "query_type": "general_info",
            "complexity": "moderate",
            "information_needs": ["medical information"],
            "requires_rag": True,
            "requires_web_search": False,
            "sub_questions": [query]
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                reasoning = nest_asyncio.run(autonomous_reasoning_gemini(query))
                return reasoning
            except Exception as e:
                logger.error(f"Error in nested async reasoning: {e}")
        else:
            reasoning = loop.run_until_complete(autonomous_reasoning_gemini(query))
            return reasoning
    except Exception as e:
        logger.error(f"Gemini MCP reasoning error: {e}")
    
    # Fallback reasoning only if all attempts failed
    logger.warning("⚠️ Falling back to default reasoning")
    return {
        "query_type": "general_info",
        "complexity": "moderate",
        "information_needs": ["medical information"],
        "requires_rag": True,
        "requires_web_search": False,
        "sub_questions": [query]
    }

def create_execution_plan(reasoning: dict, query: str, has_rag_index: bool) -> dict:
    """
    Planning: Create multi-step execution plan based on reasoning analysis.
    Returns execution plan with steps and strategy.
    """
    plan = {
        "steps": [],
        "strategy": "sequential",
        "iterations": 1
    }
    
    # Determine execution strategy
    if reasoning["complexity"] in ["complex", "multi_faceted"]:
        plan["strategy"] = "iterative"
        plan["iterations"] = 2
    
    # Step 1: Language detection and translation
    plan["steps"].append({
        "step": 1,
        "action": "detect_language",
        "description": "Detect query language and translate if needed"
    })
    
    # Step 2: RAG retrieval (if needed and available)
    if reasoning.get("requires_rag", True) and has_rag_index:
        plan["steps"].append({
            "step": 2,
            "action": "rag_retrieval",
            "description": "Retrieve relevant document context",
            "parameters": {"top_k": 15, "merge_threshold": 0.5}
        })
    
    # Step 3: Web search (if needed)
    if reasoning.get("requires_web_search", False):
        plan["steps"].append({
            "step": 3,
            "action": "web_search",
            "description": "Search web for current/updated information",
            "parameters": {"max_results": 5}
        })
    
    # Step 4: Sub-question processing (if complex)
    if reasoning.get("sub_questions") and len(reasoning["sub_questions"]) > 1:
        plan["steps"].append({
            "step": 4,
            "action": "multi_step_reasoning",
            "description": "Process sub-questions iteratively",
            "sub_questions": reasoning["sub_questions"]
        })
    
    # Step 5: Synthesis and answer generation
    plan["steps"].append({
        "step": len(plan["steps"]) + 1,
        "action": "synthesize_answer",
        "description": "Generate comprehensive answer from all sources"
    })
    
    # Step 6: Self-reflection (for complex queries)
    if reasoning["complexity"] in ["complex", "multi_faceted"]:
        plan["steps"].append({
            "step": len(plan["steps"]) + 1,
            "action": "self_reflection",
            "description": "Evaluate answer quality and completeness"
        })
    
    logger.info(f"Execution plan created: {len(plan['steps'])} steps")
    return plan

def autonomous_execution_strategy(reasoning: dict, plan: dict, use_rag: bool, use_web_search: bool, has_rag_index: bool) -> dict:
    """
    Autonomous execution: Make decisions on information gathering strategy.
    Only suggests web search override, but respects user's RAG disable setting.
    """
    strategy = {
        "use_rag": use_rag,  # Respect user's RAG setting
        "use_web_search": use_web_search,
        "reasoning_override": False,
        "rationale": ""
    }
    
    # Respect user toggle; just log recommendation if web search is disabled
    if reasoning.get("requires_web_search", False) and not use_web_search:
        strategy["rationale"] = "Reasoning suggests web search for current information, but the user kept it disabled."
    
    # Note: We don't override RAG setting because:
    # 1. User may have explicitly disabled it
    # 2. RAG requires documents to be uploaded
    # 3. We should respect user's explicit choice
    
    if strategy["rationale"]:
        logger.info(f"Autonomous reasoning note: {strategy['rationale']}")
    
    return strategy

async def gemini_supervisor_breakdown_async(query: str, use_rag: bool, use_web_search: bool, time_elapsed: float, max_duration: int = 120) -> dict:
    """
    Gemini Supervisor: Break user query into sub-topics (flexible number, explore different approaches)
    This is the main supervisor function that orchestrates the MAC architecture.
    All internal thoughts are logged, not displayed.
    """
    remaining_time = max(15, max_duration - time_elapsed)
    
    mode_description = []
    if use_rag:
        mode_description.append("RAG mode enabled - will use retrieved documents")
    if use_web_search:
        mode_description.append("Web search mode enabled - will search online sources")
    if not mode_description:
        mode_description.append("Direct answer mode - no additional context")
    
    # Calculate reasonable max topics based on time remaining
    # Allow more subtasks if we have time, but be flexible
    estimated_time_per_task = 8  # seconds per task
    max_topics_by_time = max(2, int((remaining_time - 20) / estimated_time_per_task))
    max_topics = min(max_topics_by_time, 10)  # Cap at 10, but allow more than 4
    
    prompt = f"""You are a supervisor agent coordinating with a MedSwin medical specialist model.
Break the following medical query into focused sub-topics that MedSwin can answer sequentially.
Explore different potential approaches to comprehensively address the topic.

Query: "{query}"
Mode: {', '.join(mode_description)}
Time Remaining: ~{remaining_time:.1f}s
Maximum Topics: {max_topics} (adjust based on complexity - use as many as needed for thorough coverage)

Return ONLY valid JSON (no markdown, no tables, no explanations):
{{
  "sub_topics": [
    {{
      "id": 1,
      "topic": "concise topic name",
      "instruction": "specific directive for MedSwin to answer this topic",
      "expected_tokens": 200,
      "priority": "high|medium|low",
      "approach": "brief description of approach/angle for this topic"
    }},
    ...
  ],
  "strategy": "brief strategy description explaining the breakdown approach",
  "exploration_note": "brief note on different approaches explored"
}}

Guidelines:
- Break down the query into as many subtasks as needed for comprehensive coverage
- Explore different angles/approaches (e.g., clinical, diagnostic, treatment, prevention, research perspectives)
- Each topic should be focused and answerable in ~200 tokens by MedSwin
- Prioritize topics by importance (high priority first)
- Don't limit yourself to 4 topics - use more if the query is complex or multi-faceted"""
    
    system_prompt = "You are a medical query supervisor. Break queries into structured JSON sub-topics, exploring different approaches. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    try:
        # Extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            breakdown = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Query broken into {len(breakdown.get('sub_topics', []))} sub-topics")
            logger.debug(f"[GEMINI SUPERVISOR] Breakdown: {json.dumps(breakdown, indent=2)}")
            return breakdown
        else:
            raise ValueError("Supervisor JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Breakdown parsing failed: {exc}")
        # Fallback: simple breakdown
        breakdown = {
            "sub_topics": [
                {"id": 1, "topic": "Core Question", "instruction": "Address the main medical question", "expected_tokens": 200, "priority": "high", "approach": "direct answer"},
                {"id": 2, "topic": "Clinical Details", "instruction": "Provide key clinical insights", "expected_tokens": 200, "priority": "medium", "approach": "clinical perspective"},
            ],
            "strategy": "Sequential answer with key points",
            "exploration_note": "Fallback breakdown - basic coverage"
        }
        logger.warning(f"[GEMINI SUPERVISOR] Using fallback breakdown")
        return breakdown

async def gemini_supervisor_search_strategies_async(query: str, time_elapsed: float) -> dict:
    """
    Gemini Supervisor: In search mode, break query into 1-4 searching strategies
    Returns JSON with search strategies that will be executed with ddgs
    """
    prompt = f"""You are supervising web search for a medical query.
Break this query into 1-4 focused search strategies (each targeting 1-2 sources).

Query: "{query}"

Return ONLY valid JSON:
{{
  "search_strategies": [
    {{
      "id": 1,
      "strategy": "search query string",
      "target_sources": 1,
      "focus": "what to search for"
    }},
    ...
  ],
  "max_strategies": 4
}}

Keep strategies focused and avoid overlap."""
    
    system_prompt = "You are a search strategy supervisor. Create focused search queries. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,  # Use lite model for search planning
        temperature=0.2
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            strategies = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Created {len(strategies.get('search_strategies', []))} search strategies")
            logger.debug(f"[GEMINI SUPERVISOR] Strategies: {json.dumps(strategies, indent=2)}")
            return strategies
        else:
            raise ValueError("Search strategies JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Search strategies parsing failed: {exc}")
        return {
            "search_strategies": [
                {"id": 1, "strategy": query, "target_sources": 2, "focus": "main query"}
            ],
            "max_strategies": 1
        }

async def gemini_supervisor_rag_brainstorm_async(query: str, retrieved_docs: str, time_elapsed: float) -> dict:
    """
    Gemini Supervisor: In RAG mode, brainstorm retrieved documents into 1-4 short contexts
    These contexts will be passed to MedSwin to support decision-making
    """
    # Limit retrieved docs to avoid token overflow
    max_doc_length = 3000
    if len(retrieved_docs) > max_doc_length:
        retrieved_docs = retrieved_docs[:max_doc_length] + "..."
    
    prompt = f"""You are supervising RAG context preparation for a medical query.
Brainstorm the retrieved documents into 1-4 concise, focused contexts that MedSwin can use.

Query: "{query}"
Retrieved Documents:
{retrieved_docs}

Return ONLY valid JSON:
{{
  "contexts": [
    {{
      "id": 1,
      "context": "concise summary of relevant information (keep under 500 chars)",
      "focus": "what this context covers",
      "relevance": "high|medium|low"
    }},
    ...
  ],
  "max_contexts": 4
}}

Keep contexts brief and factual. Avoid redundancy."""
    
    system_prompt = "You are a RAG context supervisor. Summarize documents into concise contexts. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,  # Use lite model for RAG brainstorming
        temperature=0.2
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            contexts = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Brainstormed {len(contexts.get('contexts', []))} RAG contexts")
            logger.debug(f"[GEMINI SUPERVISOR] Contexts: {json.dumps(contexts, indent=2)}")
            return contexts
        else:
            raise ValueError("RAG contexts JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] RAG brainstorming parsing failed: {exc}")
        # Fallback: use retrieved docs as single context
        return {
            "contexts": [
                {"id": 1, "context": retrieved_docs[:500], "focus": "retrieved information", "relevance": "high"}
            ],
            "max_contexts": 1
        }

def gemini_supervisor_breakdown(query: str, use_rag: bool, use_web_search: bool, time_elapsed: float, max_duration: int = 120) -> dict:
    """Wrapper to obtain supervisor breakdown synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP unavailable, using fallback breakdown")
        return {
            "sub_topics": [
                {"id": 1, "topic": "Core Question", "instruction": "Address the main medical question", "expected_tokens": 200, "priority": "high", "approach": "direct answer"},
                {"id": 2, "topic": "Clinical Details", "instruction": "Provide key clinical insights", "expected_tokens": 200, "priority": "medium", "approach": "clinical perspective"},
            ],
            "strategy": "Sequential answer with key points",
            "exploration_note": "Fallback breakdown - basic coverage"
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(
                    gemini_supervisor_breakdown_async(query, use_rag, use_web_search, time_elapsed, max_duration)
                )
            except Exception as exc:
                logger.error(f"[GEMINI SUPERVISOR] Nested breakdown execution failed: {exc}")
                raise
        return loop.run_until_complete(
            gemini_supervisor_breakdown_async(query, use_rag, use_web_search, time_elapsed, max_duration)
        )
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Breakdown request failed: {exc}")
        return {
            "sub_topics": [
                {"id": 1, "topic": "Core Question", "instruction": "Address the main medical question", "expected_tokens": 200, "priority": "high", "approach": "direct answer"},
            ],
            "strategy": "Direct answer",
            "exploration_note": "Fallback breakdown - single topic"
        }

def gemini_supervisor_search_strategies(query: str, time_elapsed: float) -> dict:
    """Wrapper to obtain search strategies synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP unavailable for search strategies")
        return {
            "search_strategies": [
                {"id": 1, "strategy": query, "target_sources": 2, "focus": "main query"}
            ],
            "max_strategies": 1
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(gemini_supervisor_search_strategies_async(query, time_elapsed))
            except Exception as exc:
                logger.error(f"[GEMINI SUPERVISOR] Nested search strategies execution failed: {exc}")
                return {
                    "search_strategies": [
                        {"id": 1, "strategy": query, "target_sources": 2, "focus": "main query"}
                    ],
                    "max_strategies": 1
                }
        return loop.run_until_complete(gemini_supervisor_search_strategies_async(query, time_elapsed))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Search strategies request failed: {exc}")
        return {
            "search_strategies": [
                {"id": 1, "strategy": query, "target_sources": 2, "focus": "main query"}
            ],
            "max_strategies": 1
        }

def gemini_supervisor_rag_brainstorm(query: str, retrieved_docs: str, time_elapsed: float) -> dict:
    """Wrapper to obtain RAG brainstorm synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP unavailable for RAG brainstorm")
        return {
            "contexts": [
                {"id": 1, "context": retrieved_docs[:500], "focus": "retrieved information", "relevance": "high"}
            ],
            "max_contexts": 1
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(gemini_supervisor_rag_brainstorm_async(query, retrieved_docs, time_elapsed))
            except Exception as exc:
                logger.error(f"[GEMINI SUPERVISOR] Nested RAG brainstorm execution failed: {exc}")
                return {
                    "contexts": [
                        {"id": 1, "context": retrieved_docs[:500], "focus": "retrieved information", "relevance": "high"}
                    ],
                    "max_contexts": 1
                }
        return loop.run_until_complete(gemini_supervisor_rag_brainstorm_async(query, retrieved_docs, time_elapsed))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] RAG brainstorm request failed: {exc}")
        return {
            "contexts": [
                {"id": 1, "context": retrieved_docs[:500], "focus": "retrieved information", "relevance": "high"}
            ],
            "max_contexts": 1
        }

@spaces.GPU(max_duration=120)
def execute_medswin_task(
    medical_model_obj,
    medical_tokenizer,
    task_instruction: str,
    context: str,
    system_prompt_base: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    penalty: float
) -> str:
    """
    MedSwin Specialist: Execute a single task assigned by Gemini Supervisor
    This function is tagged with @spaces.GPU to run on GPU (ZeroGPU equivalent)
    All internal thoughts are logged, only final answer is returned
    """
    # Build task-specific prompt
    if context:
        full_prompt = f"{system_prompt_base}\n\nContext:\n{context}\n\nTask: {task_instruction}\n\nAnswer concisely with key bullet points (Markdown format, no tables):"
    else:
        full_prompt = f"{system_prompt_base}\n\nTask: {task_instruction}\n\nAnswer concisely with key bullet points (Markdown format, no tables):"
    
    messages = [{"role": "system", "content": full_prompt}]
    
    # Format prompt
    if hasattr(medical_tokenizer, 'chat_template') and medical_tokenizer.chat_template is not None:
        try:
            prompt = medical_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"[MEDSWIN] Chat template failed, using manual formatting: {e}")
            prompt = format_prompt_manually(messages, medical_tokenizer)
    else:
        prompt = format_prompt_manually(messages, medical_tokenizer)
    
    # Tokenize and generate
    inputs = medical_tokenizer(prompt, return_tensors="pt").to(medical_model_obj.device)
    
    eos_token_id = medical_tokenizer.eos_token_id or medical_tokenizer.pad_token_id
    
    with torch.no_grad():
        outputs = medical_model_obj.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, 800),  # Limit per task
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=penalty,
            do_sample=True,
            eos_token_id=eos_token_id,
            pad_token_id=medical_tokenizer.pad_token_id or eos_token_id
        )
    
    # Decode response
    response = medical_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Clean response - remove any table-like formatting, ensure Markdown bullets
    response = response.strip()
    # Remove table markers if present
    if "|" in response and "---" in response:
        logger.warning("[MEDSWIN] Detected table format, converting to Markdown bullets")
        # Simple conversion: split by lines and convert to bullets
        lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('|') and '---' not in line]
        response = '\n'.join([f"- {line}" if not line.startswith('-') else line for line in lines])
    
    logger.info(f"[MEDSWIN] Task completed: {len(response)} chars generated")
    return response

async def gemini_supervisor_synthesize_async(query: str, medswin_answers: list, rag_contexts: list, search_contexts: list, breakdown: dict) -> str:
    """
    Gemini Supervisor: Synthesize final answer from all MedSwin responses with clear context
    Provides better context to create a comprehensive, well-structured final answer
    """
    # Prepare context summary
    context_summary = ""
    if rag_contexts:
        context_summary += f"Document Context Available: {len(rag_contexts)} context(s) from uploaded documents.\n"
    if search_contexts:
        context_summary += f"Web Search Context Available: {len(search_contexts)} search result(s).\n"
    
    # Combine all MedSwin answers
    all_answers_text = "\n\n---\n\n".join([f"## {i+1}. {ans}" for i, ans in enumerate(medswin_answers)])
    
    prompt = f"""You are a supervisor agent synthesizing a comprehensive medical answer from multiple specialist responses.

Original Query: "{query}"

Context Available:
{context_summary}

MedSwin Specialist Responses (from {len(medswin_answers)} sub-topics):
{all_answers_text}

Your task:
1. Synthesize all responses into a coherent, comprehensive final answer
2. Integrate information from all sub-topics seamlessly
3. Ensure the answer directly addresses the original query
4. Maintain clinical accuracy and clarity
5. Use clear structure with appropriate headings and bullet points
6. Remove redundancy and contradictions
7. Ensure all important points from MedSwin responses are included

Return the final synthesized answer in Markdown format. Do not add meta-commentary or explanations - just provide the final answer."""
    
    system_prompt = "You are a medical answer synthesis supervisor. Create comprehensive, well-structured final answers from multiple specialist responses."
    
    result = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    return result.strip()

async def gemini_supervisor_challenge_async(query: str, current_answer: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> dict:
    """
    Gemini Supervisor: Challenge and evaluate the current answer, suggesting improvements
    Returns evaluation with suggestions for enhancement
    """
    context_info = ""
    if rag_contexts:
        context_info += f"Document contexts: {len(rag_contexts)} available.\n"
    if search_contexts:
        context_info += f"Search contexts: {len(search_contexts)} available.\n"
    
    all_answers_text = "\n\n---\n\n".join([f"## {i+1}. {ans}" for i, ans in enumerate(medswin_answers)])
    
    prompt = f"""You are a supervisor agent evaluating and challenging a medical answer for quality and completeness.

Original Query: "{query}"

Available Context:
{context_info}

MedSwin Specialist Responses:
{all_answers_text}

Current Synthesized Answer:
{current_answer[:2000]}

Evaluate this answer and provide:
1. Completeness: Does it fully address the query? What's missing?
2. Accuracy: Are there any inaccuracies or contradictions?
3. Clarity: Is it well-structured and clear?
4. Context Usage: Are document/search contexts properly utilized?
5. Improvement Suggestions: Specific ways to enhance the answer

Return ONLY valid JSON:
{{
  "is_optimal": true/false,
  "completeness_score": 0-10,
  "accuracy_score": 0-10,
  "clarity_score": 0-10,
  "missing_aspects": ["..."],
  "inaccuracies": ["..."],
  "improvement_suggestions": ["..."],
  "needs_more_context": true/false,
  "enhancement_instructions": "specific instructions for improving the answer"
}}"""
    
    system_prompt = "You are a medical answer quality evaluator. Provide honest, constructive feedback in JSON format. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            evaluation = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Challenge evaluation: optimal={evaluation.get('is_optimal', False)}, scores={evaluation.get('completeness_score', 'N/A')}/{evaluation.get('accuracy_score', 'N/A')}/{evaluation.get('clarity_score', 'N/A')}")
            return evaluation
        else:
            raise ValueError("Evaluation JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Challenge evaluation parsing failed: {exc}")
        return {
            "is_optimal": True,
            "completeness_score": 7,
            "accuracy_score": 7,
            "clarity_score": 7,
            "missing_aspects": [],
            "inaccuracies": [],
            "improvement_suggestions": [],
            "needs_more_context": False,
            "enhancement_instructions": ""
        }

async def gemini_supervisor_enhance_answer_async(query: str, current_answer: str, enhancement_instructions: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> str:
    """
    Gemini Supervisor: Enhance the answer based on challenge feedback
    """
    context_info = ""
    if rag_contexts:
        context_info += f"Document contexts: {len(rag_contexts)} available.\n"
    if search_contexts:
        context_info += f"Search contexts: {len(search_contexts)} available.\n"
    
    all_answers_text = "\n\n---\n\n".join([f"## {i+1}. {ans}" for i, ans in enumerate(medswin_answers)])
    
    prompt = f"""You are a supervisor agent enhancing a medical answer based on evaluation feedback.

Original Query: "{query}"

Available Context:
{context_info}

MedSwin Specialist Responses:
{all_answers_text}

Current Answer (to enhance):
{current_answer}

Enhancement Instructions:
{enhancement_instructions}

Create an enhanced version of the answer that:
1. Addresses all improvement suggestions
2. Fills in missing aspects
3. Corrects any inaccuracies
4. Improves clarity and structure
5. Better utilizes available context
6. Maintains all valuable information from the current answer

Return the enhanced answer in Markdown format. Do not add meta-commentary."""
    
    system_prompt = "You are a medical answer enhancement supervisor. Improve answers based on evaluation feedback while maintaining accuracy."
    
    result = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    return result.strip()

async def gemini_supervisor_check_clarity_async(query: str, answer: str, use_web_search: bool) -> dict:
    """
    Gemini Supervisor: Check if answer is unclear or supervisor is unsure (only when search mode enabled)
    Returns decision on whether to trigger additional search
    """
    if not use_web_search:
        # Only check clarity when search mode is enabled
        return {"is_unclear": False, "needs_search": False, "search_queries": []}
    
    prompt = f"""You are a supervisor agent evaluating answer clarity and completeness.

Query: "{query}"

Current Answer:
{answer[:1500]}

Evaluate:
1. Is the answer unclear or incomplete?
2. Are there gaps that web search could fill?
3. Is the supervisor (you) unsure about certain aspects?

Return ONLY valid JSON:
{{
  "is_unclear": true/false,
  "needs_search": true/false,
  "uncertainty_areas": ["..."],
  "search_queries": ["specific search queries to fill gaps"],
  "rationale": "brief explanation"
}}

Only suggest search if the answer is genuinely unclear or has significant gaps that search could address."""
    
    system_prompt = "You are a clarity evaluator. Assess if additional web search is needed. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,
        temperature=0.2
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            evaluation = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Clarity check: unclear={evaluation.get('is_unclear', False)}, needs_search={evaluation.get('needs_search', False)}")
            return evaluation
        else:
            raise ValueError("Clarity check JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Clarity check parsing failed: {exc}")
        return {"is_unclear": False, "needs_search": False, "search_queries": []}

def gemini_supervisor_synthesize(query: str, medswin_answers: list, rag_contexts: list, search_contexts: list, breakdown: dict) -> str:
    """Wrapper to synthesize answer synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP unavailable for synthesis, using simple concatenation")
        return "\n\n".join(medswin_answers)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(gemini_supervisor_synthesize_async(query, medswin_answers, rag_contexts, search_contexts, breakdown))
            except Exception as exc:
                logger.error(f"[GEMINI SUPERVISOR] Nested synthesis failed: {exc}")
                return "\n\n".join(medswin_answers)
        return loop.run_until_complete(gemini_supervisor_synthesize_async(query, medswin_answers, rag_contexts, search_contexts, breakdown))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Synthesis failed: {exc}")
        return "\n\n".join(medswin_answers)

def gemini_supervisor_challenge(query: str, current_answer: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> dict:
    """Wrapper to challenge answer synchronously"""
    if not MCP_AVAILABLE:
        return {"is_optimal": True, "completeness_score": 7, "accuracy_score": 7, "clarity_score": 7, "missing_aspects": [], "inaccuracies": [], "improvement_suggestions": [], "needs_more_context": False, "enhancement_instructions": ""}
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(gemini_supervisor_challenge_async(query, current_answer, medswin_answers, rag_contexts, search_contexts))
            except Exception as exc:
                logger.error(f"[GEMINI SUPERVISOR] Nested challenge failed: {exc}")
                return {"is_optimal": True, "completeness_score": 7, "accuracy_score": 7, "clarity_score": 7, "missing_aspects": [], "inaccuracies": [], "improvement_suggestions": [], "needs_more_context": False, "enhancement_instructions": ""}
        return loop.run_until_complete(gemini_supervisor_challenge_async(query, current_answer, medswin_answers, rag_contexts, search_contexts))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Challenge failed: {exc}")
        return {"is_optimal": True, "completeness_score": 7, "accuracy_score": 7, "clarity_score": 7, "missing_aspects": [], "inaccuracies": [], "improvement_suggestions": [], "needs_more_context": False, "enhancement_instructions": ""}

def gemini_supervisor_enhance_answer(query: str, current_answer: str, enhancement_instructions: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> str:
    """Wrapper to enhance answer synchronously"""
    if not MCP_AVAILABLE:
        return current_answer
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(gemini_supervisor_enhance_answer_async(query, current_answer, enhancement_instructions, medswin_answers, rag_contexts, search_contexts))
            except Exception as exc:
                logger.error(f"[GEMINI SUPERVISOR] Nested enhancement failed: {exc}")
                return current_answer
        return loop.run_until_complete(gemini_supervisor_enhance_answer_async(query, current_answer, enhancement_instructions, medswin_answers, rag_contexts, search_contexts))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Enhancement failed: {exc}")
        return current_answer

def gemini_supervisor_check_clarity(query: str, answer: str, use_web_search: bool) -> dict:
    """Wrapper to check clarity synchronously"""
    if not MCP_AVAILABLE or not use_web_search:
        return {"is_unclear": False, "needs_search": False, "search_queries": []}
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(gemini_supervisor_check_clarity_async(query, answer, use_web_search))
            except Exception as exc:
                logger.error(f"[GEMINI SUPERVISOR] Nested clarity check failed: {exc}")
                return {"is_unclear": False, "needs_search": False, "search_queries": []}
        return loop.run_until_complete(gemini_supervisor_check_clarity_async(query, answer, use_web_search))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Clarity check failed: {exc}")
        return {"is_unclear": False, "needs_search": False, "search_queries": []}

async def self_reflection_gemini(answer: str, query: str) -> dict:
    """Self-reflection using Gemini MCP"""
    reflection_prompt = f"""Evaluate this medical answer for quality and completeness:
Query: "{query}"
Answer: "{answer[:1000]}"
Evaluate:
1. Completeness: Does it address all aspects of the query?
2. Accuracy: Is the medical information accurate?
3. Clarity: Is it clear and well-structured?
4. Sources: Are sources cited appropriately?
5. Missing Information: What important information might be missing?
Respond in JSON:
{{
    "completeness_score": 0-10,
    "accuracy_score": 0-10,
    "clarity_score": 0-10,
    "overall_score": 0-10,
    "missing_aspects": ["..."],
    "improvement_suggestions": ["..."]
}}"""
    
    # Use concise system prompt
    system_prompt = "You are a medical answer quality evaluator. Provide honest, constructive feedback."
    
    response = await call_agent(
        user_prompt=reflection_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,  # Use full model for reflection
        temperature=0.3
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            reflection = json.loads(response[json_start:json_end])
        else:
            reflection = {"overall_score": 7, "improvement_suggestions": []}
    except:
        reflection = {"overall_score": 7, "improvement_suggestions": []}
    
    logger.info(f"Self-reflection score: {reflection.get('overall_score', 'N/A')}")
    return reflection

def self_reflection(answer: str, query: str, reasoning: dict) -> dict:
    """
    Self-reflection: Evaluate answer quality and completeness.
    Returns reflection with quality score and improvement suggestions.
    """
    if not MCP_AVAILABLE:
        logger.warning("Gemini MCP not available for reflection, using fallback")
        return {"overall_score": 7, "improvement_suggestions": []}
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                return nest_asyncio.run(self_reflection_gemini(answer, query))
            except Exception as e:
                logger.error(f"Error in nested async reflection: {e}")
        else:
            return loop.run_until_complete(self_reflection_gemini(answer, query))
    except Exception as e:
        logger.error(f"Gemini MCP reflection error: {e}")
    
    return {"overall_score": 7, "improvement_suggestions": []}

async def parse_document_gemini(file_path: str, file_extension: str) -> str:
    """Parse document using Gemini MCP"""
    if not MCP_AVAILABLE:
        return ""
    
    try:
        # Read file and encode to base64
        with open(file_path, 'rb') as f:
            file_content = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine MIME type from file extension
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.csv': 'text/csv'
        }
        mime_type = mime_type_map.get(file_extension, 'application/octet-stream')
        
        # Prepare file object for Gemini MCP (use content for base64)
        files = [{
            "content": file_content,
            "type": mime_type
        }]
        
        # Use concise system prompt
        system_prompt = "Extract all text content from the document accurately."
        user_prompt = "Extract all text content from this document. Return only the extracted text, preserving structure and formatting where possible."
        
        result = await call_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            files=files,
            model=GEMINI_MODEL_LITE,  # Use lite model for parsing
            temperature=0.2
        )
        
        return result.strip()
    except Exception as e:
        logger.error(f"Gemini document parsing error: {e}")
        return ""

def extract_text_from_document(file):
    """Extract text from document using Gemini MCP"""
    file_name = file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    # Handle text files directly
    if file_extension == '.txt':
        text = file.read().decode('utf-8')
        return text, len(text.split()), None
    
    # For PDF, Word, and other documents, use Gemini MCP
    # Save file to temporary location for processing
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Write file content to temp file
            file.seek(0)  # Reset file pointer
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        # Use Gemini MCP to parse document
        if MCP_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    try:
                        import nest_asyncio
                        text = nest_asyncio.run(parse_document_gemini(tmp_file_path, file_extension))
                    except Exception as e:
                        logger.error(f"Error in nested async document parsing: {e}")
                        text = ""
                else:
                    text = loop.run_until_complete(parse_document_gemini(tmp_file_path, file_extension))
                
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                
                if text:
                    return text, len(text.split()), None
                else:
                    return None, 0, ValueError(f"Failed to extract text from {file_extension} file using Gemini MCP")
            except Exception as e:
                logger.error(f"Gemini MCP document parsing error: {e}")
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                return None, 0, ValueError(f"Error parsing {file_extension} file: {str(e)}")
        else:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            return None, 0, ValueError(f"Gemini MCP not available. Cannot parse {file_extension} files.")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None, 0, ValueError(f"Error processing {file_extension} file: {str(e)}")

@spaces.GPU(max_duration=120)
def create_or_update_index(files, request: gr.Request):
    global global_file_info
    
    if not files:
        return "Please provide files.", ""
    
    start_time = time.time()
    user_id = request.session_hash
    save_dir = f"./{user_id}_index"
    # Initialize LlamaIndex modules
    llm = get_llm_for_rag()
    embed_model = get_or_create_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model
    file_stats = []
    new_documents = []
    
    for file in tqdm(files, desc="Processing files"):
        file_basename = os.path.basename(file.name)
        text, word_count, error = extract_text_from_document(file)
        if error:
            logger.error(f"Error processing file {file_basename}: {str(error)}")
            file_stats.append({
                "name": file_basename,
                "words": 0,
                "status": f"error: {str(error)}"
            })
            continue
        
        doc = LlamaDocument(
            text=text,
            metadata={
                "file_name": file_basename,
                "word_count": word_count,
                "source": "user_upload"
            }
        )
        new_documents.append(doc)
        
        file_stats.append({
            "name": file_basename,
            "words": word_count,
            "status": "processed"
        })
        
        global_file_info[file_basename] = {
            "word_count": word_count,
            "processed_at": time.time()
        }
    
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],  
        chunk_overlap=20         
    )
    logger.info(f"Parsing {len(new_documents)} documents into hierarchical nodes")
    new_nodes = node_parser.get_nodes_from_documents(new_documents)
    new_leaf_nodes = get_leaf_nodes(new_nodes)
    new_root_nodes = get_root_nodes(new_nodes)
    logger.info(f"Generated {len(new_nodes)} total nodes ({len(new_root_nodes)} root, {len(new_leaf_nodes)} leaf)")
    
    if os.path.exists(save_dir):
        logger.info(f"Loading existing index from {save_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=save_dir)
        index = load_index_from_storage(storage_context, settings=Settings)
        docstore = storage_context.docstore
        
        docstore.add_documents(new_nodes)
        for node in tqdm(new_leaf_nodes, desc="Adding leaf nodes to index"):
            index.insert_nodes([node])
            
        total_docs = len(docstore.docs)
        logger.info(f"Updated index with {len(new_nodes)} new nodes from {len(new_documents)} files")
    else:
        logger.info("Creating new index")
        docstore = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(docstore=docstore)
        docstore.add_documents(new_nodes)
        
        index = VectorStoreIndex(
            new_leaf_nodes, 
            storage_context=storage_context, 
            settings=Settings
        )
        total_docs = len(new_documents)
        logger.info(f"Created new index with {len(new_nodes)} nodes from {len(new_documents)} files")
    
    index.storage_context.persist(persist_dir=save_dir)
    # custom outputs after processing files
    file_list_html = "<div class='file-list'>"
    for stat in file_stats:
        status_color = "#4CAF50" if stat["status"] == "processed" else "#f44336"
        file_list_html += f"<div><span style='color:{status_color}'>●</span> {stat['name']} - {stat['words']} words</div>"
    file_list_html += "</div>"
    processing_time = time.time() - start_time
    stats_output = f"<div class='stats-box'>"
    stats_output += f"✓ Processed {len(files)} files in {processing_time:.2f} seconds<br>"
    stats_output += f"✓ Created {len(new_nodes)} nodes ({len(new_leaf_nodes)} leaf nodes)<br>"
    stats_output += f"✓ Total documents in index: {total_docs}<br>"
    stats_output += f"✓ Index saved to: {save_dir}<br>"
    stats_output += "</div>"
    output_container = f"<div class='info-container'>"
    output_container += file_list_html
    output_container += stats_output
    output_container += "</div>"
    return f"Successfully indexed {len(files)} files.", output_container

@spaces.GPU(max_duration=120)
def stream_chat(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    penalty: float,
    retriever_k: int,
    merge_threshold: float,
    use_rag: bool,
    medical_model: str,
    use_web_search: bool,
    disable_agentic_reasoning: bool,
    show_thoughts: bool,
    request: gr.Request
):
    if not request:
        yield history + [{"role": "assistant", "content": "Session initialization failed. Please refresh the page."}], ""
        return
    
    # Set up thought capture handler if show_thoughts is enabled
    thought_handler = None
    if show_thoughts:
        thought_handler = ThoughtCaptureHandler()
        thought_handler.setLevel(logging.INFO)
        thought_handler.clear()  # Start fresh
        logger.addHandler(thought_handler)
    
    session_start = time.time()
    soft_timeout = 100
    hard_timeout = 118  # stop slightly before HF max duration (120s)
    
    def elapsed():
        return time.time() - session_start
    
    user_id = request.session_hash
    index_dir = f"./{user_id}_index"
    has_rag_index = os.path.exists(index_dir)
    
    # ===== MAC ARCHITECTURE: GEMINI SUPERVISOR + MEDSWIN SPECIALIST =====
    # All internal thoughts are logged, only final answer is displayed
    
    original_lang = detect_language(message)
    original_message = message
    needs_translation = original_lang != "en"
    
    pipeline_diagnostics = {
        "reasoning": None,
        "plan": None,
        "strategy_decisions": [],
        "stage_metrics": {},
        "search": {"strategies": [], "total_results": 0}
    }

    def record_stage(stage_name: str, start_time: float):
        pipeline_diagnostics["stage_metrics"][stage_name] = round(time.time() - start_time, 3)
    
    translation_stage_start = time.time()
    if needs_translation:
        logger.info(f"[GEMINI SUPERVISOR] Detected non-English language: {original_lang}, translating...")
        message = translate_text(message, target_lang="en", source_lang=original_lang)
        logger.info(f"[GEMINI SUPERVISOR] Translated query: {message[:100]}...")
    record_stage("translation", translation_stage_start)
    
    # Determine final modes (respect user settings and availability)
    final_use_rag = use_rag and has_rag_index and not disable_agentic_reasoning
    final_use_web_search = use_web_search and not disable_agentic_reasoning
    
    plan = None
    if not disable_agentic_reasoning:
        reasoning_stage_start = time.time()
        reasoning = autonomous_reasoning(message, history)
        record_stage("autonomous_reasoning", reasoning_stage_start)
        pipeline_diagnostics["reasoning"] = reasoning
        plan = create_execution_plan(reasoning, message, has_rag_index)
        pipeline_diagnostics["plan"] = plan
        execution_strategy = autonomous_execution_strategy(
            reasoning, plan, final_use_rag, final_use_web_search, has_rag_index
        )
        
        if final_use_rag and not reasoning.get("requires_rag", True):
            final_use_rag = False
            pipeline_diagnostics["strategy_decisions"].append("Skipped RAG per autonomous reasoning")
        elif not final_use_rag and reasoning.get("requires_rag", True) and not has_rag_index:
            pipeline_diagnostics["strategy_decisions"].append("Reasoning wanted RAG but no index available")
        
        if final_use_web_search and not reasoning.get("requires_web_search", False):
            final_use_web_search = False
            pipeline_diagnostics["strategy_decisions"].append("Skipped web search per autonomous reasoning")
        elif not final_use_web_search and reasoning.get("requires_web_search", False):
            if not use_web_search:
                pipeline_diagnostics["strategy_decisions"].append("User disabled web search despite reasoning request")
            else:
                pipeline_diagnostics["strategy_decisions"].append("Web search requested by reasoning but disabled by mode")
    else:
        pipeline_diagnostics["strategy_decisions"].append("Agentic reasoning disabled by user")
    
    # ===== STEP 1: GEMINI SUPERVISOR - Break query into sub-topics =====
    if disable_agentic_reasoning:
        logger.info("[MAC] Agentic reasoning disabled - using MedSwin alone")
        # Simple breakdown for direct mode
        breakdown = {
            "sub_topics": [
                {"id": 1, "topic": "Answer", "instruction": message, "expected_tokens": 400, "priority": "high", "approach": "direct answer"}
            ],
            "strategy": "Direct answer",
            "exploration_note": "Direct mode - no breakdown"
        }
    else:
        logger.info("[GEMINI SUPERVISOR] Breaking query into sub-topics...")
        breakdown = gemini_supervisor_breakdown(message, final_use_rag, final_use_web_search, elapsed(), max_duration=120)
        logger.info(f"[GEMINI SUPERVISOR] Created {len(breakdown.get('sub_topics', []))} sub-topics")
    
    # ===== STEP 2: GEMINI SUPERVISOR - Handle Search Mode =====
    search_contexts = []
    web_urls = []
    if final_use_web_search:
        search_stage_start = time.time()
        logger.info("[GEMINI SUPERVISOR] Search mode: Creating search strategies...")
        search_strategies = gemini_supervisor_search_strategies(message, elapsed())
        
        # Execute searches for each strategy
        all_search_results = []
        strategy_jobs = []
        for strategy in search_strategies.get("search_strategies", [])[:4]:  # Max 4 strategies
            search_query = strategy.get("strategy", message)
            target_sources = strategy.get("target_sources", 2)
            strategy_jobs.append({
                "query": search_query,
                "target_sources": target_sources,
                "meta": strategy
            })
        
        def execute_search(job):
            job_start = time.time()
            try:
                results = search_web(job["query"], max_results=job["target_sources"])
                duration = time.time() - job_start
                return results, duration, None
            except Exception as exc:
                return [], time.time() - job_start, exc
        
        def record_search_diag(job, duration, results_count, error=None):
            entry = {
                "query": job["query"],
                "target_sources": job["target_sources"],
                "duration": round(duration, 3),
                "results": results_count
            }
            if error:
                entry["error"] = str(error)
            pipeline_diagnostics["search"]["strategies"].append(entry)
        
        if strategy_jobs:
            max_workers = min(len(strategy_jobs), 4)
            if len(strategy_jobs) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {executor.submit(execute_search, job): job for job in strategy_jobs}
                    for future in concurrent.futures.as_completed(future_map):
                        job = future_map[future]
                        try:
                            results, duration, error = future.result()
                        except Exception as exc:
                            results, duration, error = [], 0.0, exc
                        record_search_diag(job, duration, len(results), error)
                        if not error and results:
                            all_search_results.extend(results)
                            web_urls.extend([r.get('url', '') for r in results if r.get('url')])
            else:
                job = strategy_jobs[0]
                results, duration, error = execute_search(job)
                record_search_diag(job, duration, len(results), error)
                if not error and results:
                    all_search_results.extend(results)
                    web_urls.extend([r.get('url', '') for r in results if r.get('url')])
        else:
            pipeline_diagnostics["strategy_decisions"].append("No viable web search strategies returned")
        
        pipeline_diagnostics["search"]["total_results"] = len(all_search_results)
        
        # Summarize search results with Gemini
        if all_search_results:
            logger.info(f"[GEMINI SUPERVISOR] Summarizing {len(all_search_results)} search results...")
            search_summary = summarize_web_content(all_search_results, message)
            if search_summary:
                search_contexts.append(search_summary)
                logger.info(f"[GEMINI SUPERVISOR] Search summary created: {len(search_summary)} chars")
        record_stage("web_search", search_stage_start)
    
    # ===== STEP 3: GEMINI SUPERVISOR - Handle RAG Mode =====
    rag_contexts = []
    if final_use_rag and has_rag_index:
        rag_stage_start = time.time()
        if elapsed() >= soft_timeout - 10:
            logger.warning("[GEMINI SUPERVISOR] Skipping RAG due to time pressure")
            final_use_rag = False
        else:
            logger.info("[GEMINI SUPERVISOR] RAG mode: Retrieving documents...")
            embed_model = get_or_create_embed_model()
            Settings.embed_model = embed_model
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            index = load_index_from_storage(storage_context, settings=Settings)
            base_retriever = index.as_retriever(similarity_top_k=retriever_k)
            auto_merging_retriever = AutoMergingRetriever(
                base_retriever,
                storage_context=storage_context,
                simple_ratio_thresh=merge_threshold, 
                verbose=False  # Reduce logging noise
            )
            merged_nodes = auto_merging_retriever.retrieve(message)
            retrieved_docs = "\n\n".join([n.node.text for n in merged_nodes])
            logger.info(f"[GEMINI SUPERVISOR] Retrieved {len(merged_nodes)} document nodes")
            
            # Brainstorm retrieved docs into contexts
            logger.info("[GEMINI SUPERVISOR] Brainstorming RAG contexts...")
            rag_brainstorm = gemini_supervisor_rag_brainstorm(message, retrieved_docs, elapsed())
            rag_contexts = [ctx.get("context", "") for ctx in rag_brainstorm.get("contexts", [])]
            logger.info(f"[GEMINI SUPERVISOR] Created {len(rag_contexts)} RAG contexts")
        record_stage("rag_retrieval", rag_stage_start)
    
    # ===== STEP 4: MEDSWIN SPECIALIST - Execute tasks sequentially =====
    # Initialize medical model
    medical_model_obj, medical_tokenizer = initialize_medical_model(medical_model)
    
    # Base system prompt for MedSwin (clean, no internal thoughts)
    base_system_prompt = system_prompt if system_prompt else "As a medical specialist, provide clinical and concise answers. Use Markdown format with bullet points. Do not use tables."
    
    # Prepare context for MedSwin (combine RAG and search contexts)
    combined_context = ""
    if rag_contexts:
        combined_context += "Document Context:\n" + "\n\n".join(rag_contexts[:4])  # Max 4 contexts
    if search_contexts:
        if combined_context:
            combined_context += "\n\n"
        combined_context += "Web Search Context:\n" + "\n\n".join(search_contexts)
    
    # Execute MedSwin tasks for each sub-topic
    logger.info(f"[MEDSWIN] Executing {len(breakdown.get('sub_topics', []))} tasks sequentially...")
    medswin_answers = []
    
    updated_history = history + [
        {"role": "user", "content": original_message},
        {"role": "assistant", "content": ""}
    ]
    thoughts_text = thought_handler.get_thoughts() if thought_handler else ""
    yield updated_history, thoughts_text
    
    medswin_stage_start = time.time()
    for idx, sub_topic in enumerate(breakdown.get("sub_topics", []), 1):
        if elapsed() >= hard_timeout - 5:
            logger.warning(f"[MEDSWIN] Time limit approaching, stopping at task {idx}")
            break
        
        task_instruction = sub_topic.get("instruction", "")
        topic_name = sub_topic.get("topic", f"Topic {idx}")
        priority = sub_topic.get("priority", "medium")
        
        logger.info(f"[MEDSWIN] Executing task {idx}/{len(breakdown.get('sub_topics', []))}: {topic_name} (priority: {priority})")
        
        # Select relevant context for this task (if multiple contexts available)
        task_context = combined_context
        if len(rag_contexts) > 1 and idx <= len(rag_contexts):
            # Use corresponding RAG context if available
            task_context = rag_contexts[idx - 1] if idx <= len(rag_contexts) else combined_context
        
        # Execute MedSwin task (with GPU tag)
        try:
            task_answer = execute_medswin_task(
                medical_model_obj=medical_model_obj,
                medical_tokenizer=medical_tokenizer,
                task_instruction=task_instruction,
                context=task_context if task_context else "",
                system_prompt_base=base_system_prompt,
                temperature=temperature,
                max_new_tokens=min(max_new_tokens, 800),  # Limit per task
                top_p=top_p,
                top_k=top_k,
                penalty=penalty
            )
            
            # Format task answer with topic header 
            formatted_answer = f"## {topic_name}\n\n{task_answer}"
            medswin_answers.append(formatted_answer)
            logger.info(f"[MEDSWIN] Task {idx} completed: {len(task_answer)} chars")
            
            # Stream partial answer as we complete each task
            partial_final = "\n\n".join(medswin_answers)
            updated_history[-1]["content"] = partial_final
            thoughts_text = thought_handler.get_thoughts() if thought_handler else ""
            yield updated_history, thoughts_text
    
        except Exception as e:
            logger.error(f"[MEDSWIN] Task {idx} failed: {e}")
            # Continue with next task
            continue
    record_stage("medswin_tasks", medswin_stage_start)
    
    # ===== STEP 5: GEMINI SUPERVISOR - Synthesize final answer with clear context =====
    logger.info("[GEMINI SUPERVISOR] Synthesizing final answer from all MedSwin responses...")
    raw_medswin_answers = [ans.split('\n\n', 1)[1] if '\n\n' in ans else ans for ans in medswin_answers]  # Remove headers for synthesis
    synthesis_stage_start = time.time()
    final_answer = gemini_supervisor_synthesize(message, raw_medswin_answers, rag_contexts, search_contexts, breakdown)
    record_stage("synthesis", synthesis_stage_start)
    
    if not final_answer or len(final_answer.strip()) < 50:
        # Fallback to simple concatenation if synthesis fails
        logger.warning("[GEMINI SUPERVISOR] Synthesis failed or too short, using concatenation")
        final_answer = "\n\n".join(medswin_answers) if medswin_answers else "I apologize, but I was unable to generate a response."
    
    # Clean final answer - ensure no tables, only Markdown bullets
    if "|" in final_answer and "---" in final_answer:
        logger.warning("[MEDSWIN] Final answer contains tables, converting to bullets")
        lines = final_answer.split('\n')
        cleaned_lines = []
        for line in lines:
            if '|' in line and '---' not in line:
                # Convert table row to bullet points
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    cleaned_lines.append(f"- {' / '.join(cells)}")
            elif '---' not in line:
                cleaned_lines.append(line)
        final_answer = '\n'.join(cleaned_lines)
    
    # ===== STEP 6: GEMINI SUPERVISOR - Challenge and enhance answer iteratively =====
    max_challenge_iterations = 2  # Limit iterations to avoid timeout
    challenge_iteration = 0
    challenge_stage_start = time.time()
    
    while challenge_iteration < max_challenge_iterations and elapsed() < soft_timeout - 15:
        challenge_iteration += 1
        logger.info(f"[GEMINI SUPERVISOR] Challenge iteration {challenge_iteration}/{max_challenge_iterations}...")
        
        evaluation = gemini_supervisor_challenge(message, final_answer, raw_medswin_answers, rag_contexts, search_contexts)
        
        if evaluation.get("is_optimal", False):
            logger.info(f"[GEMINI SUPERVISOR] Answer confirmed optimal after {challenge_iteration} iteration(s)")
            break
        
        enhancement_instructions = evaluation.get("enhancement_instructions", "")
        if not enhancement_instructions:
            logger.info("[GEMINI SUPERVISOR] No enhancement instructions, considering answer optimal")
            break
        
        logger.info(f"[GEMINI SUPERVISOR] Enhancing answer based on feedback...")
        enhanced_answer = gemini_supervisor_enhance_answer(
            message, final_answer, enhancement_instructions, raw_medswin_answers, rag_contexts, search_contexts
        )
        
        if enhanced_answer and len(enhanced_answer.strip()) > len(final_answer.strip()) * 0.8:  # Ensure enhancement is substantial
            final_answer = enhanced_answer
            logger.info(f"[GEMINI SUPERVISOR] Answer enhanced (new length: {len(final_answer)} chars)")
        else:
            logger.info("[GEMINI SUPERVISOR] Enhancement did not improve answer significantly, stopping")
            break
    record_stage("challenge_loop", challenge_stage_start)
    
    # ===== STEP 7: Conditional search trigger (only when search mode enabled) =====
    if final_use_web_search and elapsed() < soft_timeout - 10:
        logger.info("[GEMINI SUPERVISOR] Checking if additional search is needed...")
        clarity_stage_start = time.time()
        clarity_check = gemini_supervisor_check_clarity(message, final_answer, final_use_web_search)
        record_stage("clarity_check", clarity_stage_start)
        
        if clarity_check.get("needs_search", False) and clarity_check.get("search_queries"):
            logger.info(f"[GEMINI SUPERVISOR] Triggering additional search: {clarity_check.get('search_queries', [])}")
            additional_search_results = []
            followup_stage_start = time.time()
            for search_query in clarity_check.get("search_queries", [])[:3]:  # Limit to 3 additional searches
                if elapsed() >= soft_timeout - 5:
                    break
                extra_start = time.time()
                results = search_web(search_query, max_results=2)
                extra_duration = time.time() - extra_start
                pipeline_diagnostics["search"]["strategies"].append({
                    "query": search_query,
                    "target_sources": 2,
                    "duration": round(extra_duration, 3),
                    "results": len(results),
                    "type": "followup"
                })
                additional_search_results.extend(results)
                web_urls.extend([r.get('url', '') for r in results if r.get('url')])
            
            if additional_search_results:
                pipeline_diagnostics["search"]["total_results"] += len(additional_search_results)
                logger.info(f"[GEMINI SUPERVISOR] Summarizing {len(additional_search_results)} additional search results...")
                additional_summary = summarize_web_content(additional_search_results, message)
                if additional_summary:
                    # Enhance answer with additional search context
                    search_contexts.append(additional_summary)
                    logger.info("[GEMINI SUPERVISOR] Enhancing answer with additional search context...")
                    enhanced_with_search = gemini_supervisor_enhance_answer(
                        message, final_answer, 
                        f"Incorporate the following additional information from web search: {additional_summary}",
                        raw_medswin_answers, rag_contexts, search_contexts
                    )
                    if enhanced_with_search and len(enhanced_with_search.strip()) > 50:
                        final_answer = enhanced_with_search
                        logger.info("[GEMINI SUPERVISOR] Answer enhanced with additional search context")
            record_stage("followup_search", followup_stage_start)
    
    citations_text = ""
    
    # ===== STEP 8: Finalize answer (translate, add citations, format) =====
    # Translate back if needed
    if needs_translation and final_answer:
        logger.info(f"[GEMINI SUPERVISOR] Translating response back to {original_lang}...")
        final_answer = translate_text(final_answer, target_lang=original_lang, source_lang="en")
    
    # Add citations if web sources were used
    if web_urls:
        unique_urls = list(dict.fromkeys(web_urls))  # Preserve order, remove duplicates
        citation_links = []
        for url in unique_urls[:5]:  # Limit to 5 citations
            domain = format_url_as_domain(url)
            if domain:
                citation_links.append(f"[{domain}]({url})")
        
        if citation_links:
            citations_text = "\n\n**Sources:** " + ", ".join(citation_links)
        
    # Add speaker icon
    speaker_icon = ' 🔊'
    final_answer_with_metadata = final_answer + citations_text + speaker_icon
        
    # Update history with final answer (ONLY final answer, no internal thoughts)
    updated_history[-1]["content"] = final_answer_with_metadata
    thoughts_text = thought_handler.get_thoughts() if thought_handler else ""
    yield updated_history, thoughts_text
    
    # Clean up thought handler
    if thought_handler:
        logger.removeHandler(thought_handler)
            
    # Log completion
    diag_summary = {
        "stage_metrics": pipeline_diagnostics["stage_metrics"],
        "decisions": pipeline_diagnostics["strategy_decisions"],
        "search": pipeline_diagnostics["search"],
    }
    try:
        logger.info(f"[MAC] Diagnostics summary: {json.dumps(diag_summary)[:1200]}")
    except Exception:
        logger.info(f"[MAC] Diagnostics summary (non-serializable)")
    logger.info(f"[MAC] Final answer generated: {len(final_answer)} chars, {len(breakdown.get('sub_topics', []))} tasks completed")

def generate_speech_for_message(text: str):
    """Generate speech for a message and return audio file"""
    audio_path = generate_speech(text)
    if audio_path:
        return audio_path
    return None

def create_demo():
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(TITLE)
        gr.HTML(DESCRIPTION)
        
        with gr.Row(elem_classes="main-container"):
            with gr.Column(elem_classes="upload-section"):
                file_upload = gr.File(
                    file_count="multiple",
                    label="Drag and Drop Files Here",
                    file_types=[".pdf", ".txt", ".doc", ".docx", ".md", ".json", ".xml", ".csv"],
                    elem_id="file-upload"
                )
                upload_button = gr.Button("Upload & Index", elem_classes="upload-button")
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Upload files to start...",
                    interactive=False
                )
                file_info_output = gr.HTML(
                    label="File Information",
                    elem_classes="processing-info"
                )
                upload_button.click(
                    fn=create_or_update_index,
                    inputs=[file_upload],
                    outputs=[status_output, file_info_output]
                )
            
            with gr.Column(elem_classes="chatbot-container"):
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Chat with MedSwin... Type your question below.",
                    show_label=False,
                    type="messages"
                )
                with gr.Row(elem_classes="input-row"):
                    message_input = gr.Textbox(
                        placeholder="Type your medical question here...",
                        show_label=False,
                        container=False,
                        lines=1,
                        scale=10
                    )
                    mic_button = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="",
                        show_label=False,
                        container=False,
                        scale=1
                    )
                    submit_button = gr.Button("➤", elem_classes="submit-btn", scale=1)
                
                # Timer display for recording (shown below input row)
                recording_timer = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
                    interactive=False,
                    visible=False,
                    container=False,
                    elem_classes="recording-timer"
                    )
                
                # Handle microphone transcription
                import time
                recording_start_time = [None]
                
                def handle_recording_start():
                    """Called when recording starts"""
                    recording_start_time[0] = time.time()
                    return gr.update(visible=True, value="Recording... 0s")
                
                def handle_recording_stop(audio):
                    """Called when recording stops"""
                    recording_start_time[0] = None
                    if audio is None:
                        return gr.update(visible=False, value=""), ""
                    transcribed = transcribe_audio(audio)
                    return gr.update(visible=False, value=""), transcribed
                
                # Use JavaScript for timer updates (simpler than Gradio Timer)
                mic_button.start_recording(
                    fn=handle_recording_start,
                    outputs=[recording_timer]
                )
                
                mic_button.stop_recording(
                    fn=handle_recording_stop,
                    inputs=[mic_button],
                    outputs=[recording_timer, message_input]
                )
                
                # TTS component for generating speech from messages
                with gr.Row(visible=False) as tts_row:
                    tts_text = gr.Textbox(visible=False)
                    tts_audio = gr.Audio(label="Generated Speech", visible=False)
                
                # Function to generate speech when speaker icon is clicked
                def generate_speech_from_chat(history):
                    """Extract last assistant message and generate speech"""
                    if not history or len(history) == 0:
                        return None
                    last_msg = history[-1]
                    if last_msg.get("role") == "assistant":
                        text = last_msg.get("content", "").replace(" 🔊", "").strip()
                        if text:
                            audio_path = generate_speech(text)
                            return audio_path
                    return None
                
                # Add TTS button that appears when assistant responds
                tts_button = gr.Button("🔊 Play Response", visible=False, size="sm")
                
                # Update TTS button visibility and generate speech
                def update_tts_button(history):
                    if history and len(history) > 0 and history[-1].get("role") == "assistant":
                        return gr.update(visible=True)
                    return gr.update(visible=False)
                
                chatbot.change(
                    fn=update_tts_button,
                    inputs=[chatbot],
                    outputs=[tts_button]
                )
                
                tts_button.click(
                    fn=generate_speech_from_chat,
                    inputs=[chatbot],
                    outputs=[tts_audio]
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        disable_agentic_reasoning = gr.Checkbox(
                            value=False,
                            label="Disable agentic reasoning",
                            info="Use MedSwin model alone without agentic reasoning, RAG, or web search"
                        )
                        show_agentic_thought = gr.Button(
                            "Show agentic thought",
                            size="sm"
                        )
                    # Scrollable textbox for agentic thoughts (initially hidden)
                    agentic_thoughts_box = gr.Textbox(
                        label="Agentic Thoughts",
                        placeholder="Internal thoughts from MedSwin and supervisor will appear here...",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        visible=False,
                        elem_classes="agentic-thoughts"
                    )
                    with gr.Row():
                        use_rag = gr.Checkbox(
                            value=False,
                            label="Enable Document RAG",
                            info="Answer based on uploaded documents (upload required)"
                        )
                        use_web_search = gr.Checkbox(
                            value=False,
                            label="Enable Web Search (MCP)",
                            info="Fetch knowledge from online medical resources"
                        )
                    
                    medical_model = gr.Radio(
                        choices=list(MEDSWIN_MODELS.keys()),
                        value=DEFAULT_MEDICAL_MODEL,
                        label="Medical Model",
                        info="MedSwin TA (default), others download on first use"
                    )
                    
                    system_prompt = gr.Textbox(
                        value="As a medical specialist, provide detailed and accurate answers based on the provided medical documents and context. Ensure all information is clinically accurate and cite sources when available.",
                        label="System Prompt",
                        lines=3
                    )
                    
                    with gr.Tab("Generation Parameters"):
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.2,  
                            label="Temperature"
                        )
                        max_new_tokens = gr.Slider(
                            minimum=512,
                            maximum=4096,
                            step=128,
                            value=2048,
                            label="Max New Tokens",
                            info="Increased for medical models to prevent early stopping"
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.7, 
                            label="Top P"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,  
                            step=1,
                            value=50,  
                            label="Top K"
                        )
                        penalty = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            value=1.2,
                            label="Repetition Penalty"
                        )
                        
                    with gr.Tab("Retrieval Parameters"):
                        retriever_k = gr.Slider(
                            minimum=5,
                            maximum=30,
                            step=1,
                            value=15,
                            label="Initial Retrieval Size (Top K)"
                        )
                        merge_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            step=0.1,
                            value=0.5,
                            label="Merge Threshold (lower = more merging)"
                        )

                # Toggle function for showing/hiding agentic thoughts
                show_thoughts_state = gr.State(value=False)
                
                def toggle_thoughts_box(current_state):
                    """Toggle visibility of agentic thoughts box"""
                    new_state = not current_state
                    return gr.update(visible=new_state), new_state
                
                show_agentic_thought.click(
                    fn=toggle_thoughts_box,
                    inputs=[show_thoughts_state],
                    outputs=[agentic_thoughts_box, show_thoughts_state]
                )
                
                submit_button.click(
                    fn=stream_chat,
                    inputs=[
                        message_input, 
                        chatbot, 
                        system_prompt, 
                        temperature, 
                        max_new_tokens, 
                        top_p, 
                        top_k, 
                        penalty,
                        retriever_k,
                        merge_threshold,
                        use_rag,
                        medical_model,
                        use_web_search,
                        disable_agentic_reasoning,
                        show_thoughts_state
                    ],
                    outputs=[chatbot, agentic_thoughts_box]
                )
                
                message_input.submit(
                    fn=stream_chat,
                    inputs=[
                        message_input, 
                        chatbot, 
                        system_prompt, 
                        temperature, 
                        max_new_tokens, 
                        top_p, 
                        top_k, 
                        penalty,
                        retriever_k,
                        merge_threshold,
                        use_rag,
                        medical_model,
                        use_web_search,
                        disable_agentic_reasoning,
                        show_thoughts_state
                    ],
                    outputs=[chatbot, agentic_thoughts_box]
                )

    return demo

if __name__ == "__main__":
    # Preload models on startup
    logger.info("Preloading models on startup...")
    logger.info("Initializing default medical model (MedSwin TA)...")
    initialize_medical_model(DEFAULT_MEDICAL_MODEL)
    logger.info("Preloading TTS model...")
    try:
        initialize_tts_model()
        if global_tts_model is not None:
            logger.info("TTS model preloaded successfully!")
        else:
            logger.warning("TTS model not available - will use MCP or disable voice generation")
    except Exception as e:
        logger.warning(f"TTS model preloading failed: {e}")
        logger.warning("Text-to-speech will use MCP or be disabled")
    
    # Check Gemini MCP availability
    if MCP_AVAILABLE:
        logger.info("Gemini MCP is available for translation, summarization, document parsing, and transcription")
    else:
        logger.warning("Gemini MCP not available - translation, summarization, document parsing, and transcription features will be limited")
    
    logger.info("Model preloading complete!")
    demo = create_demo()
    demo.launch()