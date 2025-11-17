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
    """Initialize medical model (MedSwin) - download on demand"""
    global global_medical_models, global_medical_tokenizers
    if model_name not in global_medical_models or global_medical_models[model_name] is None:
        logger.info(f"Initializing medical model: {model_name}...")
        model_path = MEDSWIN_MODELS[model_name]
        # Load tokenizer with proper configuration for MedAlpaca-7B/LLaMA-based models
        # MedAlpaca-7B is finetuned from LLaMA-7B, which uses specific tokenizer settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            token=HF_TOKEN,
            use_fast=False,  # Use slow tokenizer for better compatibility with LLaMA
            padding_side="left"  # Left padding for causal LM
        )
        
        # Fix tokenizer configuration for MedSwin/MedAlpaca-based models
        # MedAlpaca-7B is based on LLaMA, which uses specific special tokens
        # LLaMA models use:
        # - bos_token: <s> (token ID 1)
        # - eos_token: </s> (token ID 2)
        # - unk_token: <unk> (token ID 0)
        # - pad_token: typically not set, but we use eos_token
        
        # Ensure eos_token is properly set (LLaMA uses </s> with ID 2)
        if tokenizer.eos_token is None:
            # Try to decode eos_token_id if it exists
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                try:
                    eos_id = tokenizer.eos_token_id
                    # Decode the token ID to get the string
                    tokenizer.eos_token = tokenizer.decode([eos_id]) if eos_id < tokenizer.vocab_size else "</s>"
                except:
                    # Fallback: LLaMA uses </s> as EOS
                    tokenizer.eos_token = "</s>"
                    try:
                        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
                    except:
                        # If </s> doesn't exist, use ID 2 (standard LLaMA EOS token ID)
                        tokenizer.eos_token_id = 2
            else:
                # Set EOS token to </s> (standard for LLaMA)
                tokenizer.eos_token = "</s>"
                try:
                    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
                except:
                    # Standard LLaMA EOS token ID is 2
                    tokenizer.eos_token_id = 2
        
        # Ensure bos_token is set (LLaMA uses <s> with ID 1)
        if tokenizer.bos_token is None:
            if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                try:
                    bos_id = tokenizer.bos_token_id
                    tokenizer.bos_token = tokenizer.decode([bos_id]) if bos_id < tokenizer.vocab_size else "<s>"
                except:
                    tokenizer.bos_token = "<s>"
                    try:
                        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
                    except:
                        tokenizer.bos_token_id = 1  # Standard LLaMA BOS token ID
            else:
                tokenizer.bos_token = "<s>"
                try:
                    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
                except:
                    tokenizer.bos_token_id = 1
        
        # Set pad_token - MedAlpaca/LLaMA models typically use EOS as PAD
        # This is important for batch processing
        if tokenizer.pad_token is None:
            # Use eos_token as pad_token (standard practice for LLaMA-based models)
            if tokenizer.eos_token is not None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                # Fallback: add a pad token (shouldn't happen if eos_token is set correctly)
                logger.warning("EOS token not set, adding [PAD] token as fallback")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        
        # Ensure tokenizer has proper attributes for LLaMA models
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
            # LLaMA models typically have 2048 or 4096 max length
            tokenizer.model_max_length = 2048
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        
        # Update model config to match tokenizer settings
        # This ensures consistency between tokenizer and model
        if hasattr(model.config, 'pad_token_id'):
            if model.config.pad_token_id != tokenizer.pad_token_id:
                model.config.pad_token_id = tokenizer.pad_token_id
        else:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        if hasattr(model.config, 'eos_token_id'):
            if model.config.eos_token_id != tokenizer.eos_token_id:
                model.config.eos_token_id = tokenizer.eos_token_id
        else:
            model.config.eos_token_id = tokenizer.eos_token_id
        
        if hasattr(model.config, 'bos_token_id'):
            if model.config.bos_token_id != tokenizer.bos_token_id:
                model.config.bos_token_id = tokenizer.bos_token_id
        else:
            model.config.bos_token_id = tokenizer.bos_token_id
        
        # Resize model embeddings if we added new tokens (shouldn't happen with proper config)
        if len(tokenizer) > model.config.vocab_size:
            logger.info(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        global_medical_models[model_name] = model
        global_medical_tokenizers[model_name] = tokenizer
        logger.info(f"Medical model {model_name} initialized successfully")
        logger.info(f"Tokenizer config: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
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
    Generate text with MedSwin model - GPU only
    
    This function only performs the actual model inference on GPU.
    All other operations (prompt preparation, post-processing) should be done outside.
    
    Note: This function is NOT decorated with @spaces.GPU because it's called from a thread
    with unpicklable objects (threading.Event, TextIteratorStreamer). The model is already
    on GPU (initialized with device_map="auto"), so the decorator is not needed.
    """
    # Tokenize prompt (this is a CPU operation but happens here for simplicity)
    # The actual GPU work is in model.generate()
    inputs = medical_tokenizer(prompt, return_tensors="pt").to(medical_model_obj.device)
    
    # Prepare generation kwargs
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=penalty,
        do_sample=True,
        stopping_criteria=stopping_criteria,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id
    )
    
    # Run generation on GPU - this is the only GPU operation
    medical_model_obj.generate(**generation_kwargs)

