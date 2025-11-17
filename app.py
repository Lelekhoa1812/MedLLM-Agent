import gradio as gr
import os
import base64
import logging
import torch
import threading
import time
import json
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
# Import GPU-tagged model functions
from model import (
    get_llm_for_rag as get_llm_for_rag_gpu,
    get_embedding_model as get_embedding_model_gpu,
    generate_with_medswin,
    initialize_medical_model,
    global_medical_models,
    global_medical_tokenizers
)
from tqdm import tqdm
from langdetect import detect, LangDetectException
# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()  # Allow nested event loops
    except ImportError:
        pass  # nest_asyncio is optional
    MCP_AVAILABLE = True
except ImportError:
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()

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

# Global model storage - models are stored in model.py
# Import the global model storage from model.py
global_file_info = {}
global_tts_model = None

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
        return None
    
    # Check if session exists and is still valid
    if global_mcp_session is not None:
        try:
            # Test if session is still alive by listing tools
            await global_mcp_session.list_tools()
            return global_mcp_session
        except Exception as e:
            logger.debug(f"Existing MCP session invalid, recreating: {e}")
            # Clean up old session
            try:
                if global_mcp_session is not None:
                    await global_mcp_session.__aexit__(None, None, None)
            except:
                pass
            try:
                if global_mcp_stdio_ctx is not None:
                    await global_mcp_stdio_ctx.__aexit__(None, None, None)
            except:
                pass
            global_mcp_session = None
            global_mcp_stdio_ctx = None
    
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
        
        logger.info(f"Creating MCP client session with command: {MCP_SERVER_COMMAND} {MCP_SERVER_ARGS}")
        server_params = StdioServerParameters(
            command=MCP_SERVER_COMMAND,
            args=MCP_SERVER_ARGS,
            env=mcp_env
        )
        
        # Correct MCP SDK usage: stdio_client is an async context manager
        # that yields (read, write) streams
        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        
        # Wait for the server process to fully start
        # The server needs time to: start Python, import modules, initialize Gemini client, start MCP server
        logger.info("⏳ Waiting for MCP server process to start...")
        # Increase wait time and add progressive checks
        for wait_attempt in range(5):
            await asyncio.sleep(1.0)  # Check every second
            # Try to peek at the read stream to see if server is responding
            # (This is a simple check - the actual initialization will happen below)
            try:
                # Check if the process is still alive by attempting a small read with timeout
                # Note: This is a best-effort check
                pass
            except:
                pass
        logger.info("⏳ MCP server startup wait complete, proceeding with initialization...")
        
        # Create ClientSession from the streams
        # ClientSession handles initialization automatically when used as context manager
        # Use the session as a context manager to ensure proper initialization
        logger.info("🔄 Creating MCP client session...")
        try:
            from mcp.types import ClientInfo
            try:
                client_info = ClientInfo(
                    name="medllm-agent",
                    version="1.0.0"
                )
                session = ClientSession(read, write, client_info=client_info)
            except (TypeError, ValueError):
                # Fallback if ClientInfo parameters are incorrect
                session = ClientSession(read, write)
        except (ImportError, AttributeError):
            # Fallback if ClientInfo is not available
            session = ClientSession(read, write)
        
        # Initialize the session using context manager pattern
        # This properly handles the initialization handshake
        logger.info("🔄 Initializing MCP session...")
        try:
            # Enter the session context - this triggers initialization
            await session.__aenter__()
            logger.info("✅ MCP session initialized, verifying tools...")
        except Exception as e:
            logger.error(f"❌ MCP session initialization failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Clean up and return None
            try:
                await stdio_ctx.__aexit__(None, None, None)
            except:
                pass
            return None
        
        # Wait for the server to be fully ready after initialization
        await asyncio.sleep(1.0)  # Wait after initialization
        
        # Verify the session works by listing tools with retries
        # This confirms the server is ready to handle requests
        max_init_retries = 15  # Increased retries
        tools_listed = False
        tools = None
        last_error = None
        for init_attempt in range(max_init_retries):
            try:
                tools = await session.list_tools()
                if tools and hasattr(tools, 'tools') and len(tools.tools) > 0:
                    logger.info(f"✅ MCP server ready with {len(tools.tools)} tools: {[t.name for t in tools.tools]}")
                    tools_listed = True
                    break
                elif tools and hasattr(tools, 'tools'):
                    # Empty tools list - might be a server issue
                    logger.warning(f"MCP server returned empty tools list (attempt {init_attempt + 1}/{max_init_retries})")
                    if init_attempt < max_init_retries - 1:
                        await asyncio.sleep(1.5)  # Slightly longer wait
                        continue
                else:
                    # Invalid response format
                    logger.warning(f"MCP server returned invalid tools response (attempt {init_attempt + 1}/{max_init_retries})")
                    if init_attempt < max_init_retries - 1:
                        await asyncio.sleep(1.5)
                        continue
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                error_msg = str(e)
                
                # Log the actual error for debugging
                if init_attempt == 0:
                    logger.debug(f"First list_tools attempt failed: {error_msg}")
                elif init_attempt < 3:
                    logger.debug(f"list_tools attempt {init_attempt + 1} failed: {error_msg}")
                
                # Handle different error types
                if "initialization" in error_str or "before initialization" in error_str or "not initialized" in error_str:
                    if init_attempt < max_init_retries - 1:
                        wait_time = 0.5 * (init_attempt + 1)  # Progressive wait: 0.5s, 1s, 1.5s...
                        logger.debug(f"Server still initializing (attempt {init_attempt + 1}/{max_init_retries}), waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                elif "invalid request" in error_str or "invalid request parameters" in error_str:
                    # Invalid request might mean the server isn't ready yet or there's a protocol issue
                    if init_attempt < max_init_retries - 1:
                        wait_time = 1.0 * (init_attempt + 1)  # Longer wait for invalid request errors
                        logger.debug(f"Invalid request error (attempt {init_attempt + 1}/{max_init_retries}), waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed - log detailed error
                        logger.error(f"❌ Invalid request parameters error persists. This may indicate a protocol mismatch.")
                        import traceback
                        logger.debug(traceback.format_exc())
                elif init_attempt < max_init_retries - 1:
                    wait_time = 0.5 * (init_attempt + 1)
                    logger.debug(f"Tool listing attempt {init_attempt + 1}/{max_init_retries} failed: {error_msg}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Could not list tools after {max_init_retries} attempts. Last error: {error_msg}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # Don't continue - if we can't list tools, the session is not usable
                    try:
                        await session.__aexit__(None, None, None)
                    except:
                        pass
                    try:
                        await stdio_ctx.__aexit__(None, None, None)
                    except:
                        pass
                    return None
        
        if not tools_listed:
            error_msg = str(last_error) if last_error else "Unknown error"
            logger.error(f"MCP server failed to initialize - tools could not be listed. Last error: {error_msg}")
            try:
                await session.__aexit__(None, None, None)
            except:
                pass
            try:
                await stdio_ctx.__aexit__(None, None, None)
            except:
                pass
            return None
        
        # Store both the session and stdio context to keep them alive
        global_mcp_session = session
        global_mcp_stdio_ctx = stdio_ctx
        logger.info("MCP client session created successfully")
        return session
    except Exception as e:
        logger.error(f"Failed to create MCP client session: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        global_mcp_session = None
        global_mcp_stdio_ctx = None
        return None

async def call_agent(user_prompt: str, system_prompt: str = None, files: list = None, model: str = None, temperature: float = 0.2) -> str:
    """
    Call Gemini MCP generate_content tool via MCP protocol.
    
    This function uses the MCP (Model Context Protocol) to call Gemini AI,
    NOT direct API calls. It connects to the bundled agent.py MCP server
    which provides the generate_content tool.
    
    Used for: translation, summarization, document parsing, transcription, reasoning
    """
    if not MCP_AVAILABLE:
        logger.warning("MCP not available for Gemini call")
        return ""
    
    try:
        session = await get_mcp_session()
        if session is None:
            logger.warning("Failed to get MCP session for Gemini call")
            return ""
        
        # Retry listing tools if it fails the first time
        # Use more retries and longer waits since MCP server might need time
        max_retries = 5
        tools = None
        for attempt in range(max_retries):
            try:
                tools = await session.list_tools()
                if tools and hasattr(tools, 'tools') and len(tools.tools) > 0:
                    break
                else:
                    raise ValueError("Empty tools list")
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 1.0 * (attempt + 1)  # Progressive wait
                    logger.debug(f"Failed to list tools (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to list MCP tools after {max_retries} attempts: {e}")
                    return ""
        
        if not tools or not hasattr(tools, 'tools'):
            logger.error("Invalid tools response from MCP server")
            return ""
        
        # Find generate_content tool
        generate_tool = None
        for tool in tools.tools:
            if tool.name == "generate_content" or "generate_content" in tool.name.lower():
                generate_tool = tool
                logger.info(f"Found Gemini MCP tool: {tool.name}")
                break
        
        if not generate_tool:
            logger.warning(f"Gemini MCP generate_content tool not found. Available tools: {[t.name for t in tools.tools]}")
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
        
        logger.info(f"🔧 [MCP] Calling Gemini MCP tool '{generate_tool.name}' for: {user_prompt[:100]}...")
        logger.info(f"📋 [MCP] Arguments: model={model}, temperature={temperature}, files={len(files) if files else 0}")
        result = await session.call_tool(generate_tool.name, arguments=arguments)
        
        # Parse result
        if hasattr(result, 'content') and result.content:
            for item in result.content:
                if hasattr(item, 'text'):
                    response_text = item.text.strip()
                    logger.info(f"✅ [MCP] Gemini MCP returned response ({len(response_text)} chars)")
                    return response_text
        logger.warning("⚠️ [MCP] Gemini MCP returned empty or invalid result")
        return ""
    except Exception as e:
        logger.error(f"Gemini MCP call error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return ""

# initialize_medical_model is now imported from model.py


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
        import traceback
        logger.debug(traceback.format_exc())
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
        logger.debug(f"MCP TTS error: {e}")
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
                    logger.info("Generated speech via MCP")
                    return audio_path
        except Exception as e:
            logger.debug(f"MCP TTS not available: {e}")
    
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
    """Manually format prompt for MedSwin/MedAlpaca-based models without chat template
    
    MedSwin is finetuned from MedAlpaca-7B, which uses the Alpaca instruction format:
    ### Instruction:
    {instruction}
    ### Input:
    {input}  (optional, can be empty)
    ### Response:
    {response}
    """
    # Combine system and user messages into instruction and input
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
    
    # Format for MedSwin/MedAlpaca-based models
    # MedAlpaca format requires Instruction, Input (optional), and Response sections
    if system_content and user_content:
        # Both system and user content: system is instruction, user is input
        instruction = system_content.strip()
        input_text = user_content.strip()
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    elif system_content:
        # Only system content: use as instruction, empty input
        instruction = system_content.strip()
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n\n### Response:\n"
    elif user_content:
        # Only user content: use as instruction, empty input
        instruction = user_content.strip()
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n\n### Response:\n"
    else:
        # Fallback: empty prompt
        prompt = "### Instruction:\n\n### Input:\n\n### Response:\n"
    
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
    """
    Search web using MCP web search tool (e.g., DuckDuckGo MCP server).
    
    This function uses MCP tools for web search, NOT direct API calls.
    It looks for MCP tools with names containing "search", "duckduckgo", "ddg", or "web".
    """
    if not MCP_AVAILABLE:
        return []
    
    try:
        session = await get_mcp_session()
        if session is None:
            return []
        
        # Retry listing tools if it fails the first time
        max_retries = 3
        tools = None
        for attempt in range(max_retries):
            try:
                tools = await session.list_tools()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"Failed to list MCP tools after {max_retries} attempts: {e}")
                    return []
        
        if not tools or not hasattr(tools, 'tools'):
            return []
        
        # Look for web search tools (DuckDuckGo, search, etc.)
        search_tool = None
        for tool in tools.tools:
            tool_name_lower = tool.name.lower()
            if any(keyword in tool_name_lower for keyword in ["search", "duckduckgo", "ddg", "web"]):
                search_tool = tool
                logger.info(f"Found web search MCP tool: {tool.name}")
                break
        
        if search_tool:
            try:
                logger.info(f"🔍 [MCP] Using web search MCP tool '{search_tool.name}' for: {query[:100]}...")
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
                    logger.info(f"✅ [MCP] Web search MCP tool returned {len(web_content)} results")
                    return web_content
            except Exception as e:
                logger.error(f"Error calling web search MCP tool: {e}")
        
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
    logger.info(f"📝 [MCP] Summarizing {len(content_list)} web search results using Gemini MCP...")
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
    
    if result:
        logger.info(f"✅ [MCP] Web content summarized successfully using Gemini MCP ({len(result)} chars)")
    else:
        logger.warning("⚠️ [MCP] Gemini MCP summarization returned empty result")
    
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

# get_llm_for_rag is now imported from model.py as get_llm_for_rag_gpu

async def autonomous_reasoning_gemini(query: str) -> dict:
    """Autonomous reasoning using Gemini MCP"""
    logger.info(f"🧠 [MCP] Analyzing query with Gemini MCP: {query[:100]}...")
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
        logger.info("🤔 [MCP] Using Gemini MCP for autonomous reasoning...")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                reasoning = nest_asyncio.run(autonomous_reasoning_gemini(query))
                if reasoning and reasoning.get("query_type") != "general_info":  # Check if we got real reasoning
                    logger.info(f"✅ [MCP] Gemini MCP reasoning successful: {reasoning.get('query_type')}, complexity: {reasoning.get('complexity')}")
                    return reasoning
                else:
                    logger.warning("⚠️ [MCP] Gemini MCP returned fallback reasoning, using it anyway")
                    return reasoning
            except Exception as e:
                logger.error(f"❌ Error in nested async reasoning: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            reasoning = loop.run_until_complete(autonomous_reasoning_gemini(query))
            if reasoning and reasoning.get("query_type") != "general_info":
                logger.info(f"✅ [MCP] Gemini MCP reasoning successful: {reasoning.get('query_type')}, complexity: {reasoning.get('complexity')}")
                return reasoning
            else:
                logger.warning("⚠️ [MCP] Gemini MCP returned fallback reasoning, using it anyway")
                return reasoning
    except Exception as e:
        logger.error(f"❌ Gemini MCP reasoning error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
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
    
    # Only suggest web search override (RAG requires documents, so we respect user's choice)
    if reasoning.get("requires_web_search", False) and not use_web_search:
        strategy["use_web_search"] = True
        strategy["reasoning_override"] = True
        strategy["rationale"] += "Reasoning suggests web search for current information. "
    
    # Note: We don't override RAG setting because:
    # 1. User may have explicitly disabled it
    # 2. RAG requires documents to be uploaded
    # 3. We should respect user's explicit choice
    
    if strategy["reasoning_override"]:
        logger.info(f"Autonomous override: {strategy['rationale']}")
    
    return strategy

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
        import traceback
        logger.debug(traceback.format_exc())
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

def create_or_update_index(files, request: gr.Request):
    global global_file_info
    
    if not files:
        return "Please provide files.", ""
    
    start_time = time.time()
    user_id = request.session_hash
    save_dir = f"./{user_id}_index"
    # Initialize LlamaIndex modules - use GPU functions for model inference only
    llm = get_llm_for_rag_gpu()
    embed_model = get_embedding_model_gpu()
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
    request: gr.Request
):
    if not request:
        yield history + [{"role": "assistant", "content": "Session initialization failed. Please refresh the page."}]
        return
    
    user_id = request.session_hash
    index_dir = f"./{user_id}_index"
    has_rag_index = os.path.exists(index_dir)
    
    # ===== AUTONOMOUS REASONING =====
    logger.info("🤔 Starting autonomous reasoning...")
    reasoning = autonomous_reasoning(message, history)
    
    # ===== PLANNING =====
    logger.info("📋 Creating execution plan...")
    plan = create_execution_plan(reasoning, message, has_rag_index)
    
    # ===== AUTONOMOUS EXECUTION STRATEGY =====
    logger.info("🎯 Determining execution strategy...")
    execution_strategy = autonomous_execution_strategy(reasoning, plan, use_rag, use_web_search, has_rag_index)
    
    # Use autonomous strategy decisions (respect user's RAG setting)
    final_use_rag = execution_strategy["use_rag"] and has_rag_index  # Only use RAG if enabled AND documents exist
    final_use_web_search = execution_strategy["use_web_search"]
    
    # Show reasoning override message if applicable
    reasoning_note = ""
    if execution_strategy["reasoning_override"]:
        reasoning_note = f"\n\n💡 *Autonomous Reasoning: {execution_strategy['rationale']}*"
    
    # Detect language and translate if needed (Step 1 of plan)
    original_lang = detect_language(message)
    original_message = message
    needs_translation = original_lang != "en"
    
    if needs_translation:
        logger.info(f"Detected non-English language: {original_lang}, translating to English...")
        message = translate_text(message, target_lang="en", source_lang=original_lang)
        logger.info(f"Translated query: {message}")
    
    # Initialize medical model
    medical_model_obj, medical_tokenizer = initialize_medical_model(medical_model)
    
    # Adjust system prompt based on RAG setting and reasoning
    if final_use_rag:
        base_system_prompt = system_prompt if system_prompt else "As a medical specialist, provide clinical and concise answers based on the provided medical documents and context."
    else:
        base_system_prompt = "As a medical specialist, provide short and concise clinical answers. Be brief and avoid lengthy explanations. Focus on key medical facts only."
    
    # Add reasoning context to system prompt for complex queries
    if reasoning["complexity"] in ["complex", "multi_faceted"]:
        base_system_prompt += f"\n\nQuery Analysis: This is a {reasoning['complexity']} {reasoning['query_type']} query. Address all sub-questions: {', '.join(reasoning.get('sub_questions', [])[:3])}"
    
    # ===== EXECUTION: RAG Retrieval (Step 2) =====
    rag_context = ""
    source_info = ""
    if final_use_rag and has_rag_index:
        # Use GPU function for embedding model
        embed_model = get_embedding_model_gpu()
        Settings.embed_model = embed_model
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context, settings=Settings)
        base_retriever = index.as_retriever(similarity_top_k=retriever_k)
        auto_merging_retriever = AutoMergingRetriever(
            base_retriever,
            storage_context=storage_context,
            simple_ratio_thresh=merge_threshold, 
            verbose=True
        )
        logger.info(f"Query: {message}")
        retrieval_start = time.time()
        merged_nodes = auto_merging_retriever.retrieve(message)
        logger.info(f"Retrieved {len(merged_nodes)} merged nodes in {time.time() - retrieval_start:.2f}s")
        merged_file_sources = {}
        for node in merged_nodes:
            if hasattr(node.node, 'metadata') and 'file_name' in node.node.metadata:
                file_name = node.node.metadata['file_name']
                if file_name not in merged_file_sources:
                    merged_file_sources[file_name] = 0
                merged_file_sources[file_name] += 1
        logger.info(f"Merged retrieval file distribution: {merged_file_sources}")
        rag_context = "\n\n".join([n.node.text for n in merged_nodes])
        if merged_file_sources:
            source_info = "\n\nRetrieved information from files: " + ", ".join(merged_file_sources.keys())
    
    # ===== EXECUTION: Web Search (Step 3) =====
    web_context = ""
    web_sources = []
    web_urls = []  # Store URLs for citations
    if final_use_web_search:
        logger.info("🌐 Performing web search (using MCP tools, with Gemini MCP for summarization)...")
        # search_web() tries MCP web search tool first, then falls back to direct API
        web_results = search_web(message, max_results=5)
        if web_results:
            logger.info(f"📊 Found {len(web_results)} web search results, now summarizing with Gemini MCP...")
            # summarize_web_content() uses Gemini MCP via call_agent()
            web_summary = summarize_web_content(web_results, message)
            if web_summary and len(web_summary) > 50:  # Check if we got a real summary
                logger.info(f"✅ [MCP] Gemini MCP summarization successful ({len(web_summary)} chars)")
                web_context = f"\n\nAdditional Web Sources (summarized with Gemini MCP):\n{web_summary}"
            else:
                logger.warning("⚠️ [MCP] Gemini MCP summarization failed or returned empty, using raw results")
                # Fallback: use first result's content
                web_context = f"\n\nAdditional Web Sources:\n{web_results[0].get('content', '')[:500]}"
            web_sources = [r['title'] for r in web_results[:3]]
            # Extract unique URLs for citations
            web_urls = [r.get('url', '') for r in web_results if r.get('url')]
            logger.info(f"✅ Web search completed: {len(web_results)} results, summarized with Gemini MCP")
        else:
            logger.warning("⚠️ Web search returned no results")
    
    # Build final context
    context_parts = []
    if rag_context:
        context_parts.append(f"Document Context:\n{rag_context}")
    if web_context:
        context_parts.append(web_context)
    
    full_context = "\n\n".join(context_parts) if context_parts else ""
    
    # Build system prompt
    if final_use_rag or final_use_web_search:
        formatted_system_prompt = f"{base_system_prompt}\n\n{full_context}{source_info}"
    else:
        formatted_system_prompt = base_system_prompt
    
    # Prepare messages
    messages = [{"role": "system", "content": formatted_system_prompt}]
    for entry in history:
        messages.append(entry)
    messages.append({"role": "user", "content": message})
    
    # Get EOS token and adjust stopping criteria
    eos_token_id = medical_tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = medical_tokenizer.pad_token_id
    
    # Increase max tokens for medical models (prevent early stopping)
    max_new_tokens = int(max_new_tokens) if isinstance(max_new_tokens, (int, float)) else 2048
    max_new_tokens = max(max_new_tokens, 1024)  # Minimum 1024 tokens for medical answers
    
    # Check if tokenizer has chat template, otherwise format manually
    if hasattr(medical_tokenizer, 'chat_template') and medical_tokenizer.chat_template is not None:
        try:
            prompt = medical_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Chat template failed, using manual formatting: {e}")
            # Fallback to manual formatting
            prompt = format_prompt_manually(messages, medical_tokenizer)
    else:
        # Manual formatting for models without chat template
        prompt = format_prompt_manually(messages, medical_tokenizer)
    
    inputs = medical_tokenizer(prompt, return_tensors="pt").to(medical_model_obj.device)
    prompt_length = inputs['input_ids'].shape[1]
    
    stop_event = threading.Event()
    
    class StopOnEvent(StoppingCriteria):
        def __init__(self, stop_event):
            super().__init__()
            self.stop_event = stop_event

        def __call__(self, input_ids, scores, **kwargs):
            return self.stop_event.is_set()
    
    # Custom stopping criteria that doesn't stop on EOS too early
    class MedicalStoppingCriteria(StoppingCriteria):
        def __init__(self, eos_token_id, prompt_length, min_new_tokens=100):
            super().__init__()
            self.eos_token_id = eos_token_id
            self.prompt_length = prompt_length
            self.min_new_tokens = min_new_tokens

        def __call__(self, input_ids, scores, **kwargs):
            current_length = input_ids.shape[1]
            new_tokens = current_length - self.prompt_length
            last_token = input_ids[0, -1].item()
            
            # Don't stop on EOS if we haven't generated enough new tokens
            if new_tokens < self.min_new_tokens:
                return False
            # Allow EOS after minimum new tokens have been generated
            return last_token == self.eos_token_id
    
    stopping_criteria = StoppingCriteriaList([
        StopOnEvent(stop_event),
        MedicalStoppingCriteria(eos_token_id, prompt_length, min_new_tokens=100)
    ])
    
    streamer = TextIteratorStreamer(
        medical_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    temperature = float(temperature) if isinstance(temperature, (int, float)) else 0.7
    top_p = float(top_p) if isinstance(top_p, (int, float)) else 0.95
    top_k = int(top_k) if isinstance(top_k, (int, float)) else 50
    penalty = float(penalty) if isinstance(penalty, (int, float)) else 1.2
    
    # Call GPU function for model inference only
    thread = threading.Thread(
        target=generate_with_medswin,
        kwargs={
            "medical_model_obj": medical_model_obj,
            "medical_tokenizer": medical_tokenizer,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "penalty": penalty,
            "eos_token_id": eos_token_id,
            "pad_token_id": medical_tokenizer.pad_token_id or eos_token_id,
            "stop_event": stop_event,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria
        }
    )
    thread.start()
    
    updated_history = history + [
        {"role": "user", "content": original_message},
        {"role": "assistant", "content": ""}
    ]
    yield updated_history
    
    partial_response = ""
    try:
        for new_text in streamer:
            partial_response += new_text
            updated_history[-1]["content"] = partial_response
            yield updated_history
        
        # ===== SELF-REFLECTION (Step 6) =====
        if reasoning["complexity"] in ["complex", "multi_faceted"]:
            logger.info("🔍 Performing self-reflection on answer quality...")
            reflection = self_reflection(partial_response, message, reasoning)
            
            # Add reflection note if score is low or improvements suggested
            if reflection.get("overall_score", 10) < 7 or reflection.get("improvement_suggestions"):
                reflection_note = f"\n\n---\n**Self-Reflection** (Score: {reflection.get('overall_score', 'N/A')}/10)"
                if reflection.get("improvement_suggestions"):
                    reflection_note += f"\n💡 Suggestions: {', '.join(reflection['improvement_suggestions'][:2])}"
                partial_response += reflection_note
                updated_history[-1]["content"] = partial_response
        
        # Add reasoning note if autonomous override occurred
        if reasoning_note:
            partial_response = reasoning_note + "\n\n" + partial_response
            updated_history[-1]["content"] = partial_response
        
        # Translate back if needed
        if needs_translation and partial_response:
            logger.info(f"Translating response back to {original_lang}...")
            translated_response = translate_text(partial_response, target_lang=original_lang, source_lang="en")
            partial_response = translated_response
        
        # Add citations if web sources were used
        citations_text = ""
        if web_urls:
            # Get unique domains
            unique_urls = list(dict.fromkeys(web_urls))  # Preserve order, remove duplicates
            citation_links = []
            for url in unique_urls[:5]:  # Limit to 5 citations
                domain = format_url_as_domain(url)
                if domain:
                    # Create markdown link: [domain](url)
                    citation_links.append(f"[{domain}]({url})")
            
            if citation_links:
                citations_text = "\n\n**Sources:** " + ", ".join(citation_links)
        
        # Add speaker icon and citations to assistant message
        speaker_icon = ' 🔊'
        partial_response_with_speaker = partial_response + citations_text + speaker_icon
        updated_history[-1]["content"] = partial_response_with_speaker
        
        yield updated_history
            
    except GeneratorExit:
        stop_event.set()
        thread.join()
        raise

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
                        use_rag = gr.Checkbox(
                            value=False,
                            label="Enable Document RAG",
                            info="Answer based on uploaded documents (requires document upload)"
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
                        use_web_search
                    ],
                    outputs=chatbot
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
                        use_web_search
                    ],
                    outputs=chatbot
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
