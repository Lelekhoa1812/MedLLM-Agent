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
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    penalty: float,
    eos_token_id: int,
    pad_token_id: int,
    prompt_length: int,
    min_new_tokens: int = 100
):
    """
    Internal GPU function that only takes picklable arguments.
    This function is decorated with @spaces.GPU and creates streamer/stopping criteria internally.
    
    Returns: TextIteratorStreamer that can be consumed by the caller
    """
    # Get model and tokenizer from global storage (already loaded)
    medical_model_obj = global_medical_models.get(model_name)
    medical_tokenizer = global_medical_tokenizers.get(model_name)
    
    if medical_model_obj is None or medical_tokenizer is None:
        raise ValueError(f"Model {model_name} not initialized. Call initialize_medical_model first.")
    
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
    
    # Create streamer inside GPU function (can't be pickled, so create here)
    streamer = TextIteratorStreamer(
        medical_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=None
    )
    
    # Create stopping criteria inside GPU function (can't be pickled)
    # Use a simple flag-based stopping instead of threading.Event
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
    # Start generation in a separate thread so we can return the streamer immediately
    def run_generation():
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
    
    # Start generation in background thread
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()
    
    # Return streamer so caller can consume it
    return streamer


@spaces.GPU(max_duration=120)
def generate_with_medswin_gpu(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    penalty: float,
    eos_token_id: int,
    pad_token_id: int,
    prompt_length: int,
    min_new_tokens: int = 100
):
    """
    GPU-decorated wrapper that only takes picklable arguments.
    This function is called by generate_with_medswin which handles unpicklable objects.
    """
    return _generate_with_medswin_internal(
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        penalty=penalty,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        prompt_length=prompt_length,
        min_new_tokens=min_new_tokens
    )


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
    Public API function that maintains backward compatibility.
    This function is NOT decorated with @spaces.GPU to avoid pickling issues.
    It calls the GPU-decorated function internally.
    
    Note: stop_event and the original streamer/stopping_criteria are kept for API compatibility
    but the actual generation uses new objects created inside the GPU function.
    """
    # Get model name from global storage (find which model this is)
    model_name = None
    for name, model in global_medical_models.items():
        if model is medical_model_obj:
            model_name = name
            break
    
    if model_name is None:
        raise ValueError("Model not found in global storage. Ensure model is initialized via initialize_medical_model.")
    
    # Calculate prompt length for stopping criteria
    inputs = medical_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False
    )
    prompt_length = inputs['input_ids'].shape[1]
    
    # Call GPU function with only picklable arguments
    # The GPU function will create its own streamer and stopping criteria
    gpu_streamer = generate_with_medswin_gpu(
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        penalty=penalty,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        prompt_length=prompt_length,
        min_new_tokens=100
    )
    
    # Copy tokens from GPU streamer to the original streamer
    # TextIteratorStreamer uses a queue internally (usually named 'queue' or '_queue')
    # We need to read from GPU streamer and write to the original streamer's queue
    def copy_stream():
        try:
            # Find the queue in the original streamer
            streamer_queue = None
            if hasattr(streamer, 'queue'):
                streamer_queue = streamer.queue
            elif hasattr(streamer, '_queue'):
                streamer_queue = streamer._queue
            else:
                # Try to get queue from tokenizer's queue if available
                logger.warning("Could not find streamer queue attribute, trying alternative method")
                # TextIteratorStreamer might store queue differently - check all attributes
                for attr in dir(streamer):
                    if 'queue' in attr.lower() and not attr.startswith('__'):
                        try:
                            streamer_queue = getattr(streamer, attr)
                            if hasattr(streamer_queue, 'put'):
                                break
                        except:
                            pass
            
            if streamer_queue is None:
                logger.error("Could not access streamer queue - tokens will be lost!")
                return
            
            # Read tokens from GPU streamer and put them into original streamer's queue
            for token in gpu_streamer:
                streamer_queue.put(token)
            
            # Signal end of stream (TextIteratorStreamer uses None or StopIteration)
            try:
                streamer_queue.put(None)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error copying stream: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Start copying in background
    copy_thread = threading.Thread(target=copy_stream, daemon=True)
    copy_thread.start()
    
    # Return immediately - caller will consume from original streamer
    return

