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
    """Initialize medical model (MedSwin) - download on demand
    
    Following standard MedAlpaca/LLaMA initialization pattern:
    - Simple tokenizer loading without over-complication
    - Model loading with device_map="auto" for ZeroGPU Spaces
    - Proper pad_token setup for LLaMA-based models
    - Float16 for memory efficiency
    - Ensure tokenizer padding side is set correctly
    """
    global global_medical_models, global_medical_tokenizers
    
    if model_name not in global_medical_models or global_medical_models[model_name] is None:
        logger.info(f"Initializing medical model: {model_name}...")
        model_path = MEDSWIN_MODELS[model_name]
        
        # Load tokenizer - simple and clean, following example pattern
        # Use fast tokenizer if available (default), fallback to slow if needed
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                token=HF_TOKEN,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load fast tokenizer, trying slow tokenizer: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                token=HF_TOKEN,
                use_fast=False,
                trust_remote_code=True
            )
        
        # LLaMA models don't have pad_token by default, set it to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Set padding side to left for generation (LLaMA models expect this)
        tokenizer.padding_side = "left"
        
        # Ensure tokenizer is properly configured
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
            tokenizer.model_max_length = 4096
        
        # Load model - use device_map="auto" for ZeroGPU Spaces
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        
        # Ensure model is in eval mode
        model.eval()
        
        global_medical_models[model_name] = model
        global_medical_tokenizers[model_name] = tokenizer
        logger.info(f"Medical model {model_name} initialized successfully")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        logger.info(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        logger.info(f"Tokenizer padding side: {tokenizer.padding_side}")
    
    return global_medical_models[model_name], global_medical_tokenizers[model_name]


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


def get_embedding_model():
    """Get embedding model for RAG - GPU only"""
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, token=HF_TOKEN)

def _generate_with_medswin_internal(
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
    prompt_length: int,
    min_new_tokens: int = 100,
    streamer: TextIteratorStreamer = None,
    stopping_criteria: StoppingCriteriaList = None
):
    """
    Internal generation function that runs directly on GPU.
    Model is already on GPU via device_map="auto", so no @spaces.GPU decorator needed.
    This avoids pickling issues with streamer and stopping_criteria.
    """
    # Ensure model is in evaluation mode
    medical_model_obj.eval()
    
    # Get device - handle device_map="auto" case
    device = next(medical_model_obj.parameters()).device
    
    # Tokenize prompt - CRITICAL: use consistent tokenization settings
    # For LLaMA-based models, the tokenizer automatically adds BOS token
    inputs = medical_tokenizer(
        prompt, 
        return_tensors="pt",
        add_special_tokens=True,  # Let tokenizer add BOS/EOS as needed
        padding=False,  # No padding for single sequence generation
        truncation=False  # Don't truncate - let model handle length
    ).to(device)
    
    # Log tokenization info for debugging
    actual_prompt_length = inputs['input_ids'].shape[1]
    logger.info(f"Tokenized prompt: {actual_prompt_length} tokens on device {device}")
    
    # Use provided streamer and stopping_criteria (created in caller to avoid pickling)
    if streamer is None:
        streamer = TextIteratorStreamer(
            medical_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None
        )
    
    if stopping_criteria is None:
        # Create simple stopping criteria if not provided
        class SimpleStoppingCriteria(StoppingCriteria):
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
            SimpleStoppingCriteria(eos_token_id, actual_prompt_length, min_new_tokens)
        ])
    
    # Prepare generation kwargs - following standard MedAlpaca/LLaMA pattern
    # Ensure all parameters are valid and within expected ranges
    generation_kwargs = {
        **inputs,  # Unpack input_ids and attention_mask
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": max(0.01, min(temperature, 2.0)),  # Clamp temperature to valid range
        "top_p": max(0.0, min(top_p, 1.0)),  # Clamp top_p to valid range
        "top_k": max(1, int(top_k)),  # Ensure top_k is at least 1
        "repetition_penalty": max(1.0, min(penalty, 2.0)),  # Clamp repetition_penalty
        "do_sample": True,
        "stopping_criteria": stopping_criteria,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id
    }
    
    # Validate token IDs are valid
    if eos_token_id is None or eos_token_id < 0:
        logger.warning(f"Invalid EOS token ID: {eos_token_id}, using tokenizer default")
        eos_token_id = medical_tokenizer.eos_token_id or medical_tokenizer.pad_token_id
        generation_kwargs["eos_token_id"] = eos_token_id
    
    if pad_token_id is None or pad_token_id < 0:
        logger.warning(f"Invalid PAD token ID: {pad_token_id}, using EOS token")
        pad_token_id = eos_token_id
        generation_kwargs["pad_token_id"] = pad_token_id
    
    # Run generation on GPU with torch.no_grad() for efficiency
    # Model is already on GPU, so this will run on GPU automatically
    with torch.no_grad():
        try:
            logger.debug(f"Starting generation with max_new_tokens={max_new_tokens}, temperature={generation_kwargs['temperature']}, top_p={generation_kwargs['top_p']}, top_k={generation_kwargs['top_k']}")
            logger.debug(f"EOS token ID: {eos_token_id}, PAD token ID: {pad_token_id}")
            medical_model_obj.generate(**generation_kwargs)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


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
    Public API function for model generation.
    
    This function is NOT decorated with @spaces.GPU because:
    1. The model is already on GPU via device_map="auto" during initialization
    2. Generation will automatically run on GPU where the model is located
    3. This avoids pickling issues with streamer, stop_event, and stopping_criteria
    
    The @spaces.GPU decorator is only needed for model loading, which is handled
    separately in initialize_medical_model (though that also doesn't need it since
    device_map="auto" handles GPU placement).
    """
    # Calculate prompt length for stopping criteria (if not already calculated)
    inputs = medical_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False
    )
    prompt_length = inputs['input_ids'].shape[1]
    
    # Call internal generation function directly
    # Model is already on GPU, so generation will happen on GPU automatically
    _generate_with_medswin_internal(
        medical_model_obj=medical_model_obj,
        medical_tokenizer=medical_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        penalty=penalty,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        prompt_length=prompt_length,
        min_new_tokens=100,
        streamer=streamer,  # Use the provided streamer (created in caller)
        stopping_criteria=stopping_criteria  # Use the provided stopping criteria
    )
    
    # Function returns immediately - generation happens in background via streamer
    return

