import gradio as gr
import os
import PyPDF2
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
from tqdm import tqdm
from langdetect import detect, LangDetectException
import whisper
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
TRANSLATION_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MEDSWIN_MODELS = {
    "MedSwin SFT": "MedSwin/MedSwin-7B-SFT",
    "MedSwin KD": "MedSwin/MedSwin-7B-KD",
    "MedSwin TA": "MedSwin/MedSwin-Merged-TA-SFT-0.7"
}
DEFAULT_MEDICAL_MODEL = "MedSwin SFT"
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"  # Domain-tuned medical embedding model
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
TTS_MODEL = "maya-research/maya1"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

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
global_translation_model = None
global_translation_tokenizer = None
global_medical_models = {}
global_medical_tokenizers = {}
global_file_info = {}
global_whisper_model = None
global_tts_model = None

# MCP client storage
global_mcp_client = None
global_mcp_context = None  # Store the context manager
# MCP server configuration via environment variables
# Example: MCP_SERVER_COMMAND="python" MCP_SERVER_ARGS="-m duckduckgo_mcp_server"
# Or: MCP_SERVER_COMMAND="npx" MCP_SERVER_ARGS="-y @modelcontextprotocol/server-duckduckgo"
MCP_SERVER_COMMAND = os.environ.get("MCP_SERVER_COMMAND", "python")
MCP_SERVER_ARGS = os.environ.get("MCP_SERVER_ARGS", "-m duckduckgo_mcp_server").split() if os.environ.get("MCP_SERVER_ARGS") else ["-m", "duckduckgo_mcp_server"]

def initialize_translation_model():
    """Initialize DeepSeek-R1 model for translation purposes"""
    global global_translation_model, global_translation_tokenizer
    if global_translation_model is None or global_translation_tokenizer is None:
        logger.info("Initializing translation model (DeepSeek-R1-8B)...")
        global_translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL, token=HF_TOKEN)
        global_translation_model = AutoModelForCausalLM.from_pretrained(
            TRANSLATION_MODEL,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        logger.info("Translation model initialized successfully")

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

def initialize_whisper_model():
    """Initialize Whisper model for speech-to-text"""
    global global_whisper_model
    if global_whisper_model is None:
        logger.info("Initializing Whisper model for speech transcription...")
        try:
            # Check if we're in a spaces environment (has spaces patching)
            in_spaces_env = hasattr(torch, '_spaces_patched') or 'spaces' in str(type(torch.Tensor.to))
            
            # Try loading from HuggingFace with device handling for spaces compatibility
            try:
                if in_spaces_env:
                    # In spaces environment, load on CPU and don't move to device
                    logger.info("Detected spaces environment, loading Whisper on CPU")
                    global_whisper_model = whisper.load_model("large-v3-turbo", device="cpu")
                    # Don't move to GPU in spaces environment
                else:
                    # Normal environment, let whisper handle device
                    global_whisper_model = whisper.load_model("large-v3-turbo")
            except NotImplementedError as e:
                # Handle sparse tensor error from spaces library
                if "SparseTensorImpl" in str(e) or "storage" in str(e).lower():
                    logger.warning(f"Spaces library compatibility issue: {e}")
                    logger.info("Trying to load Whisper model with workaround...")
                    try:
                        # Try loading on CPU explicitly
                        global_whisper_model = whisper.load_model("base", device="cpu")
                    except Exception as e2:
                        logger.error(f"Failed to load Whisper with workaround: {e2}")
                        global_whisper_model = None
                        return None
                else:
                    raise
            except Exception as e1:
                logger.warning(f"Failed to load large-v3-turbo: {e1}")
                try:
                    # Fallback to base model with CPU
                    global_whisper_model = whisper.load_model("base", device="cpu")
                except Exception as e2:
                    logger.error(f"Failed to load Whisper base model: {e2}")
                    # Set to None to indicate failure - will use MCP or skip transcription
                    global_whisper_model = None
                    return None
        except Exception as e:
            logger.error(f"Whisper model initialization error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            global_whisper_model = None
            return None
        logger.info("Whisper model initialized successfully")
    return global_whisper_model

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

async def transcribe_audio_mcp(audio_path: str) -> str:
    """Transcribe audio using MCP Whisper tool"""
    global global_mcp_client
    
    if not MCP_AVAILABLE:
        return ""
    
    try:
        # Initialize MCP client if needed
        if global_mcp_client is None:
            return ""
        
        # Find Whisper tool
        tools = await global_mcp_client.list_tools()
        whisper_tool = None
        for tool in tools.tools:
            if "whisper" in tool.name.lower() or "transcribe" in tool.name.lower() or "speech" in tool.name.lower():
                whisper_tool = tool
                logger.info(f"Found MCP Whisper tool: {tool.name}")
                break
        
        if whisper_tool:
            result = await global_mcp_client.call_tool(
                whisper_tool.name,
                arguments={"audio_path": audio_path, "language": "en"}
            )
            
            # Parse result
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        return item.text.strip()
        return ""
    except Exception as e:
        logger.debug(f"MCP transcription error: {e}")
        return ""

def transcribe_audio(audio):
    """Transcribe audio to text using Whisper (with MCP fallback)"""
    global global_whisper_model
    
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
        
        # Try MCP first if available
        if MCP_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    try:
                        import nest_asyncio
                        transcribed = nest_asyncio.run(transcribe_audio_mcp(audio_path))
                        if transcribed:
                            logger.info(f"Transcribed via MCP: {transcribed}")
                            return transcribed
                    except:
                        pass
                else:
                    transcribed = loop.run_until_complete(transcribe_audio_mcp(audio_path))
                    if transcribed:
                        logger.info(f"Transcribed via MCP: {transcribed}")
                        return transcribed
            except Exception as e:
                logger.debug(f"MCP transcription not available: {e}")
        
        # Fallback to local Whisper model
        if global_whisper_model is None:
            initialize_whisper_model()
        
        if global_whisper_model is None:
            logger.warning("Whisper model not available and MCP not working")
            return ""
        
        # Transcribe
        result = global_whisper_model.transcribe(audio_path, language="en")
        transcribed_text = result["text"].strip()
        logger.info(f"Transcribed: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

async def generate_speech_mcp(text: str) -> str:
    """Generate speech using MCP TTS tool"""
    global global_mcp_client
    
    if not MCP_AVAILABLE:
        return None
    
    try:
        # Initialize MCP client if needed
        if global_mcp_client is None:
            return None
        
        # Find TTS tool
        tools = await global_mcp_client.list_tools()
        tts_tool = None
        for tool in tools.tools:
            if "tts" in tool.name.lower() or "speech" in tool.name.lower() or "synthesize" in tool.name.lower():
                tts_tool = tool
                logger.info(f"Found MCP TTS tool: {tool.name}")
                break
        
        if tts_tool:
            result = await global_mcp_client.call_tool(
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

async def translate_text_mcp(text: str, target_lang: str = "en", source_lang: str = None) -> str:
    """Translate text using MCP translation tool"""
    global global_mcp_client
    
    if not MCP_AVAILABLE:
        return ""
    
    try:
        # Initialize MCP client if needed
        if global_mcp_client is None:
            return ""
        
        # Find translation tool
        tools = await global_mcp_client.list_tools()
        translate_tool = None
        for tool in tools.tools:
            if "translate" in tool.name.lower() or "translation" in tool.name.lower():
                translate_tool = tool
                logger.info(f"Found MCP translation tool: {tool.name}")
                break
        
        if translate_tool:
            args = {"text": text, "target_language": target_lang}
            if source_lang:
                args["source_language"] = source_lang
            
            result = await global_mcp_client.call_tool(
                translate_tool.name,
                arguments=args
            )
            
            # Parse result
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        return item.text.strip()
        return ""
    except Exception as e:
        logger.debug(f"MCP translation error: {e}")
        return ""

def translate_text(text: str, target_lang: str = "en", source_lang: str = None) -> str:
    """Translate text using DeepSeek-R1 model (with MCP fallback)"""
    # Try MCP first if available
    if MCP_AVAILABLE:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                try:
                    import nest_asyncio
                    translated = nest_asyncio.run(translate_text_mcp(text, target_lang, source_lang))
                    if translated:
                        logger.info(f"Translated via MCP: {translated[:50]}...")
                        return translated
                except:
                    pass
            else:
                translated = loop.run_until_complete(translate_text_mcp(text, target_lang, source_lang))
                if translated:
                    logger.info(f"Translated via MCP: {translated[:50]}...")
                    return translated
        except Exception as e:
            logger.debug(f"MCP translation not available: {e}")
    
    # Fallback to local translation model
    global global_translation_model, global_translation_tokenizer
    if global_translation_model is None or global_translation_tokenizer is None:
        initialize_translation_model()
    
    if source_lang:
        prompt = f"Translate the following {source_lang} text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
    else:
        prompt = f"Translate the following text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
    
    messages = [
        {"role": "system", "content": "You are a professional translator. Translate accurately and concisely."},
        {"role": "user", "content": prompt}
    ]
    
    prompt_text = global_translation_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = global_translation_tokenizer(prompt_text, return_tensors="pt").to(global_translation_model.device)
    
    with torch.no_grad():
        outputs = global_translation_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=global_translation_tokenizer.eos_token_id
        )
    
    response = global_translation_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

async def search_web_mcp(query: str, max_results: int = 5) -> list:
    """Search web using MCP tools"""
    global global_mcp_client, global_mcp_context
    
    if not MCP_AVAILABLE:
        logger.warning("MCP not available, falling back to direct search")
        return search_web_fallback(query, max_results)
    
    try:
        # Initialize MCP client if not already initialized
        if global_mcp_client is None:
            logger.info(f"Initializing MCP client with command: {MCP_SERVER_COMMAND} {MCP_SERVER_ARGS}")
            # Try to connect to MCP server
            # This assumes MCP server is running via stdio
            server_params = StdioServerParameters(
                command=MCP_SERVER_COMMAND,
                args=MCP_SERVER_ARGS
            )
            try:
                global_mcp_client = await stdio_client(server_params)
                await global_mcp_client.__aenter__()
                logger.info("MCP client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MCP client: {e}")
                logger.info("Falling back to direct search. To use MCP, ensure MCP server is installed and configured.")
                global_mcp_client = None
                global_mcp_context = None
                return search_web_fallback(query, max_results)
        
        # Call MCP search tool
        try:
            tools = await global_mcp_client.list_tools()
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return search_web_fallback(query, max_results)
        
        search_tool = None
        for tool in tools.tools:
            if "search" in tool.name.lower() or "duckduckgo" in tool.name.lower():
                search_tool = tool
                logger.info(f"Found MCP search tool: {tool.name}")
                break
        
        if search_tool:
            # Execute search via MCP
            try:
                result = await global_mcp_client.call_tool(
                    search_tool.name,
                    arguments={"query": query, "max_results": max_results}
                )
            except Exception as e:
                logger.error(f"Error calling MCP search tool: {e}")
                return search_web_fallback(query, max_results)
            
            # Parse MCP result
            web_content = []
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        # Parse JSON response from MCP
                        import json
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
                                'content': item.text
                            })
            
            # If MCP search didn't return results, try fetching content via MCP fetch tool
            if web_content:
                # Try to fetch full content for each result using MCP fetch tool
                fetch_tool = None
                for tool in tools.tools:
                    if "fetch" in tool.name.lower() or "scrape" in tool.name.lower() or "get" in tool.name.lower():
                        fetch_tool = tool
                        logger.info(f"Found MCP fetch tool: {tool.name}")
                        break
                
                if fetch_tool:
                    for item in web_content[:3]:  # Fetch content for top 3 results
                        if not item.get('url'):
                            continue
                        try:
                            fetch_result = await global_mcp_client.call_tool(
                                fetch_tool.name,
                                arguments={"url": item['url']}
                            )
                            if hasattr(fetch_result, 'content') and fetch_result.content:
                                for content_item in fetch_result.content:
                                    if hasattr(content_item, 'text'):
                                        # Extract text content
                                        full_text = content_item.text
                                        if len(full_text) > 1000:
                                            full_text = full_text[:1000] + "..."
                                        item['content'] = item.get('content', '') + "\n" + full_text[:500]
                        except Exception as e:
                            logger.debug(f"Could not fetch content for {item['url']}: {e}")
                            continue
            
            if web_content:
                logger.info(f"MCP search returned {len(web_content)} results")
                return web_content
            else:
                logger.warning("MCP search returned no results, falling back to direct search")
                return search_web_fallback(query, max_results)
        else:
            logger.warning("MCP search tool not found, falling back to direct search")
            return search_web_fallback(query, max_results)
            
    except Exception as e:
        logger.error(f"MCP web search error: {e}, falling back to direct search")
        import traceback
        logger.debug(traceback.format_exc())
        return search_web_fallback(query, max_results)

def search_web_fallback(query: str, max_results: int = 5) -> list:
    """Fallback web search using DuckDuckGo directly (when MCP is not available)"""
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
            return web_content
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []

def search_web(query: str, max_results: int = 5) -> list:
    """Search web using MCP tools (synchronous wrapper)"""
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
                    return nest_asyncio.run(search_web_mcp(query, max_results))
                except (ImportError, AttributeError):
                    # Fallback: run in thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, search_web_mcp(query, max_results))
                        return future.result(timeout=30)
            else:
                return loop.run_until_complete(search_web_mcp(query, max_results))
        except Exception as e:
            logger.error(f"Error running async MCP search: {e}")
            return search_web_fallback(query, max_results)
    else:
        return search_web_fallback(query, max_results)

def summarize_web_content(content_list: list, query: str) -> str:
    """Summarize web search results using DeepSeek-R1 model"""
    global global_translation_model, global_translation_tokenizer
    if global_translation_model is None or global_translation_tokenizer is None:
        initialize_translation_model()
    
    combined_content = "\n\n".join([f"Source: {item['title']}\n{item['content']}" for item in content_list[:3]])
    
    prompt = f"""Summarize the following web search results related to the query: "{query}"

Extract key medical information, facts, and insights. Be concise and focus on reliable information.

Search Results:
{combined_content}

Summary:"""
    
    messages = [
        {"role": "system", "content": "You are a medical information summarizer. Extract and summarize key medical facts accurately."},
        {"role": "user", "content": prompt}
    ]
    
    prompt_text = global_translation_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = global_translation_tokenizer(prompt_text, return_tensors="pt").to(global_translation_model.device)
    
    with torch.no_grad():
        outputs = global_translation_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True,
            pad_token_id=global_translation_tokenizer.eos_token_id
        )
    
    summary = global_translation_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return summary.strip()

def get_llm_for_rag(temperature=0.7, max_new_tokens=256, top_p=0.95, top_k=50):
    """Get LLM for RAG indexing (uses translation model)"""
    if global_translation_model is None or global_translation_tokenizer is None:
        initialize_translation_model()
    
    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=max_new_tokens,
        tokenizer=global_translation_tokenizer,
        model=global_translation_model,
        generate_kwargs={
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
    )

def autonomous_reasoning(query: str, history: list) -> dict:
    """
    Autonomous reasoning: Analyze query complexity, intent, and information needs.
    Returns reasoning analysis with query type, complexity, and required information sources.
    """
    global global_translation_model, global_translation_tokenizer
    if global_translation_model is None or global_translation_tokenizer is None:
        initialize_translation_model()
    
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
    
    messages = [
        {"role": "system", "content": "You are a medical reasoning system. Analyze queries systematically and provide structured JSON responses."},
        {"role": "user", "content": reasoning_prompt}
    ]
    
    prompt_text = global_translation_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = global_translation_tokenizer(prompt_text, return_tensors="pt").to(global_translation_model.device)
    
    with torch.no_grad():
        outputs = global_translation_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=global_translation_tokenizer.eos_token_id
        )
    
    response = global_translation_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
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

def self_reflection(answer: str, query: str, reasoning: dict) -> dict:
    """
    Self-reflection: Evaluate answer quality and completeness.
    Returns reflection with quality score and improvement suggestions.
    """
    global global_translation_model, global_translation_tokenizer
    if global_translation_model is None or global_translation_tokenizer is None:
        initialize_translation_model()
    
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
    
    messages = [
        {"role": "system", "content": "You are a medical answer quality evaluator. Provide honest, constructive feedback."},
        {"role": "user", "content": reflection_prompt}
    ]
    
    prompt_text = global_translation_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = global_translation_tokenizer(prompt_text, return_tensors="pt").to(global_translation_model.device)
    
    with torch.no_grad():
        outputs = global_translation_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=global_translation_tokenizer.eos_token_id
        )
    
    response = global_translation_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    import json
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

def extract_text_from_document(file):
    file_name = file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension == '.txt':
        text = file.read().decode('utf-8')
        return text, len(text.split()), None
    elif file_extension == '.pdf':
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n\n".join(page.extract_text() for page in pdf_reader.pages)
        return text, len(text.split()), None
    else:
        return None, 0, ValueError(f"Unsupported file format: {file_extension}")

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
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)
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
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)
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
    if final_use_web_search:
        logger.info("🌐 Performing web search (MCP)...")
        web_results = search_web(message, max_results=5)
        if web_results:
            web_summary = summarize_web_content(web_results, message)
            web_context = f"\n\nAdditional Web Sources (MCP):\n{web_summary}"
            web_sources = [r['title'] for r in web_results[:3]]
            logger.info(f"Web search completed, found {len(web_results)} results")
    
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
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=penalty,
        do_sample=True,
        stopping_criteria=stopping_criteria,
        eos_token_id=eos_token_id,
        pad_token_id=medical_tokenizer.pad_token_id or eos_token_id
    )
    
    thread = threading.Thread(target=medical_model_obj.generate, kwargs=generation_kwargs)
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
        
        # Add speaker icon to assistant message
        speaker_icon = ' 🔊'
        partial_response_with_speaker = partial_response + speaker_icon
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
                    file_types=[".pdf", ".txt"],
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
                    placeholder="Chat with your medical documents here... Type your question below.",
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
                        info="MedSwin SFT (default), others download on first use"
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
    logger.info("Initializing translation model (DeepSeek-R1)...")
    try:
        initialize_translation_model()
        logger.info("Translation model (DeepSeek-R1) preloaded successfully!")
    except Exception as e:
        logger.error(f"Translation model preloading failed: {e}")
        logger.warning("Translation features may be limited")
    logger.info("Initializing default medical model (MedSwin SFT)...")
    initialize_medical_model(DEFAULT_MEDICAL_MODEL)
    logger.info("Preloading Whisper model...")
    try:
        initialize_whisper_model()
        if global_whisper_model is not None:
            logger.info("Whisper model preloaded successfully!")
        else:
            logger.warning("Whisper model not available - will use MCP or disable transcription")
    except Exception as e:
        logger.warning(f"Whisper model preloading failed: {e}")
        logger.warning("Speech-to-text will use MCP or be disabled")
        global_whisper_model = None
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
    logger.info("Model preloading complete!")
    demo = create_demo()
    demo.launch()
