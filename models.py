"""Model initialization and management"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from logger import logger
import config

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    TTS = None


def initialize_medical_model(model_name: str):
    """Initialize medical model (MedSwin) - download on demand"""
    if model_name not in config.global_medical_models or config.global_medical_models[model_name] is None:
        logger.info(f"Initializing medical model: {model_name}...")
        model_path = config.MEDSWIN_MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=config.HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            token=config.HF_TOKEN,
            torch_dtype=torch.float16
        )
        config.global_medical_models[model_name] = model
        config.global_medical_tokenizers[model_name] = tokenizer
        logger.info(f"Medical model {model_name} initialized successfully")
    return config.global_medical_models[model_name], config.global_medical_tokenizers[model_name]


def initialize_tts_model():
    """Initialize TTS model for text-to-speech"""
    if not TTS_AVAILABLE:
        logger.warning("TTS library not installed. TTS features will be disabled.")
        return None
    if config.global_tts_model is None:
        try:
            logger.info("Initializing TTS model for voice generation...")
            config.global_tts_model = TTS(model_name=config.TTS_MODEL, progress_bar=False)
            logger.info("TTS model initialized successfully")
        except Exception as e:
            logger.warning(f"TTS model initialization failed: {e}")
            logger.warning("TTS features will be disabled. If pyworld dependency is missing, try: pip install TTS --no-deps && pip install coqui-tts")
            config.global_tts_model = None
    return config.global_tts_model


def get_or_create_embed_model():
    """Reuse embedding model to avoid reloading weights each request"""
    if config.global_embed_model is None:
        logger.info("Initializing shared embedding model for RAG retrieval...")
        config.global_embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL, token=config.HF_TOKEN)
    return config.global_embed_model


def get_llm_for_rag(temperature=0.7, max_new_tokens=256, top_p=0.95, top_k=50):
    """Get LLM for RAG indexing (uses medical model)"""
    medical_model_obj, medical_tokenizer = initialize_medical_model(config.DEFAULT_MEDICAL_MODEL)
    
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

