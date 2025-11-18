"""Audio transcription and text-to-speech functions"""
import os
import asyncio
import tempfile
import soundfile as sf
from logger import logger
from mcp import MCP_AVAILABLE, call_agent, get_mcp_session, get_cached_mcp_tools
import config
from models import TTS_AVAILABLE, initialize_tts_model

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None


async def transcribe_audio_gemini(audio_path: str) -> str:
    """Transcribe audio using Gemini MCP"""
    if not MCP_AVAILABLE:
        return ""
    
    try:
        audio_path_abs = os.path.abspath(audio_path)
        files = [{"path": audio_path_abs}]
        
        system_prompt = "You are a professional transcription service. Provide accurate, well-formatted transcripts."
        user_prompt = "Please transcribe this audio file. Include speaker identification if multiple speakers are present, and format it with proper punctuation and paragraphs, remove mumble, ignore non-verbal noises."
        
        result = await call_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            files=files,
            model=config.GEMINI_MODEL_LITE,
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
        if isinstance(audio, str):
            audio_path = audio
        elif isinstance(audio, tuple):
            sample_rate, audio_data = audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, audio_data, samplerate=sample_rate)
                audio_path = tmp_file.name
        else:
            audio_path = audio
        
        if MCP_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    if nest_asyncio:
                        transcribed = nest_asyncio.run(transcribe_audio_gemini(audio_path))
                        if transcribed:
                            logger.info(f"Transcribed via Gemini MCP: {transcribed[:50]}...")
                            return transcribed
                    else:
                        logger.error("nest_asyncio not available for nested async transcription")
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
        session = await get_mcp_session()
        if session is None:
            return None
        
        tools = await get_cached_mcp_tools()
        tts_tool = None
        for tool in tools:
            tool_name_lower = tool.name.lower()
            if "tts" in tool_name_lower or "speech" in tool_name_lower or "synthesize" in tool_name_lower:
                tts_tool = tool
                logger.info(f"Found MCP TTS tool: {tool.name}")
                break
        
        if tts_tool:
            result = await session.call_tool(
                tts_tool.name,
                arguments={"text": text, "language": "en"}
            )
            
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        if os.path.exists(item.text):
                            return item.text
                    elif hasattr(item, 'data') and item.data:
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
    
    if MCP_AVAILABLE:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if nest_asyncio:
                    audio_path = nest_asyncio.run(generate_speech_mcp(text))
                    if audio_path:
                        logger.info("Generated speech via MCP")
                        return audio_path
            else:
                audio_path = loop.run_until_complete(generate_speech_mcp(text))
                if audio_path:
                    return audio_path
        except Exception as e:
            pass
    
    if not TTS_AVAILABLE:
        logger.error("TTS library not installed. Please install TTS to use voice generation.")
        return None
    
    if config.global_tts_model is None:
        initialize_tts_model()
    
    if config.global_tts_model is None:
        logger.error("TTS model not available. Please check dependencies.")
        return None
    
    try:
        wav = config.global_tts_model.tts(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, wav, samplerate=22050)
            return tmp_file.name
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

