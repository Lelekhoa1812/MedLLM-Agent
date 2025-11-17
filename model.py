"""
Model inference functions that require GPU.
These functions are tagged with @spaces.GPU(max_duration=120) to ensure
they only run on GPU and don't waste GPU time on CPU operations.
"""

import os
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import spaces
import threading

logger = logging.getLogger(__name__)

# Model configurations
MEDSWIN_MODELS = {
    "MedSwin SFT": "MedSwin/MedSwin-7B-SFT",
    "MedSwin KD": "MedSwin/MedSwin-7B-KD",
    "MedSwin TA": "MedSwin/MedSwin-Merged-TA-SFT-0.7"
}
DEFAULT_MEDICAL_MODEL = "MedSwin TA"
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Global model storage (shared with app.py)
# These will be initialized in app.py and accessed here
global_medical_models = {}
global_medical_tokenizers = {}


def initialize_medical_model(model_name: str):
    """Initialize medical model (MedSwin) - simplified configuration for MedAlpaca-based models
    
    Simplified initialization following MedAlpaca best practices:
    - Use default tokenizer configuration from model
    - Minimal configuration changes
    - Ensure proper pad_token setup for LLaMA-based models
    """
    global global_medical_models, global_medical_tokenizers
    if model_name not in global_medical_models or global_medical_models[model_name] is None:
        logger.info(f"Initializing medical model: {model_name}...")
        model_path = MEDSWIN_MODELS[model_name]
        
        # Load tokenizer - use default configuration from the model
        # MedAlpaca/MedSwin models come with proper tokenizer config
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            token=HF_TOKEN,
            padding_side="left"  # Left padding for causal LM generation
        )
        
        # Only set pad_token if it's missing (use eos_token as pad_token for LLaMA-based models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model - simplified configuration
        # Use default settings from the model, minimal overrides
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        
        # Ensure model is in eval mode (important for inference)
        model.eval()
        
        # Ensure model config matches tokenizer (only pad_token_id needs to be synced)
        if hasattr(model.config, 'pad_token_id') and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        global_medical_models[model_name] = model
        global_medical_tokenizers[model_name] = tokenizer
        logger.info(f"Medical model {model_name} initialized successfully")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Tokenizer: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}, bos_token={tokenizer.bos_token}")
        logger.info(f"Model vocab size: {len(tokenizer)}")
    return global_medical_models[model_name], global_medical_tokenizers[model_name]


@spaces.GPU(max_duration=120)
def get_llm_for_rag(temperature=0.7, max_new_tokens=256, top_p=0.95, top_k=50):
    """Get LLM for RAG indexing (uses medical model) - GPU only"""
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


@spaces.GPU(max_duration=120)
def get_embedding_model():
    """Get embedding model for RAG - GPU only"""
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)


def generate_with_medswin(
    medical_model_obj,
    medical_tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    penalty: float,
    eos_token_id: int,
    pad_token_id: int,
    stop_event: threading.Event,
    streamer: TextIteratorStreamer,
    stopping_criteria: StoppingCriteriaList
):
    """
    Generate text with MedSwin model - simplified inference for MedAlpaca-based models
    
    This function performs tokenization and model inference.
    The model is already on GPU (initialized with device_map="auto").
    
    Simplified approach following MedAlpaca best practices:
    - Ensure model is in eval mode
    - Use proper tokenization (no padding for single sequence)
    - Use standard generation parameters
    """
    # Ensure model is in evaluation mode
    medical_model_obj.eval()
    
    # Tokenize prompt - simple and clean tokenization
    # MedAlpaca models expect clean tokenization without special handling
    inputs = medical_tokenizer(
        prompt, 
        return_tensors="pt",
        padding=False,  # No padding for single sequence
        truncation=True,  # Truncate if too long (respect model max length)
        max_length=2048  # Reasonable max length for prompt
    )
    
    # Move inputs to the same device as the model
    device = next(medical_model_obj.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Prepare generation kwargs - simplified and standard for MedAlpaca models
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask", None),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": penalty,
        "do_sample": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "streamer": streamer,
        "stopping_criteria": stopping_criteria
    }
    
    # Remove None values to avoid issues
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    
    # Run generation on GPU with torch.no_grad() for efficiency
    with torch.no_grad():
        try:
            medical_model_obj.generate(**generation_kwargs)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

