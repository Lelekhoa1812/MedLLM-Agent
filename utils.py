"""Utility functions for translation, language detection, and formatting"""
import asyncio
from langdetect import detect, LangDetectException
from logger import logger
from client import MCP_AVAILABLE, call_agent
from config import GEMINI_MODEL_LITE

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None


def format_prompt_manually(messages: list, tokenizer) -> str:
    """Manually format prompt for models without chat template"""
    system_content = ""
    user_content = ""
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content
    
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
        return "en"


def format_url_as_domain(url: str) -> str:
    """Format URL as simple domain name (e.g., www.mayoclinic.org)"""
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if domain.startswith('www.'):
            return domain
        elif domain:
            return domain
        return url
    except Exception:
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
    
    system_prompt = "You are a professional translator. Translate accurately and concisely."
    
    result = await call_agent(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,
        temperature=0.2
    )
    
    return result.strip()


def translate_text(text: str, target_lang: str = "en", source_lang: str = None) -> str:
    """Translate text using Gemini MCP"""
    if not MCP_AVAILABLE:
        logger.warning("Gemini MCP not available for translation")
        return text
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                translated = nest_asyncio.run(translate_text_gemini(text, target_lang, source_lang))
                if translated:
                    logger.info(f"Translated via Gemini MCP: {translated[:50]}...")
                    return translated
            else:
                logger.error("Error in nested async translation: nest_asyncio not available")
        else:
            translated = loop.run_until_complete(translate_text_gemini(text, target_lang, source_lang))
            if translated:
                logger.info(f"Translated via Gemini MCP: {translated[:50]}...")
                return translated
    except Exception as e:
        logger.error(f"Gemini MCP translation error: {e}")
    
    return text

