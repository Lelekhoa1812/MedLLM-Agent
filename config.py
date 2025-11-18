"""Configuration constants and global storage"""
import os
import threading

# Model configurations
MEDSWIN_MODELS = {
    "MedSwin SFT": "MedSwin/MedSwin-7B-SFT",
    "MedSwin KD": "MedSwin/MedSwin-7B-KD",
    "MedSwin TA": "MedSwin/MedSwin-Merged-TA-SFT-0.7"
}
DEFAULT_MEDICAL_MODEL = "MedSwin TA"
EMBEDDING_MODEL = "abhinand/MedEmbed-large-v0.1"
TTS_MODEL = "maya-research/maya1"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

# Gemini MCP configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_LITE = os.environ.get("GEMINI_MODEL_LITE", "gemini-2.5-flash-lite")

# MCP server configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
agent_path = os.path.join(script_dir, "agent.py")
MCP_SERVER_COMMAND = os.environ.get("MCP_SERVER_COMMAND", "python")
MCP_SERVER_ARGS = os.environ.get("MCP_SERVER_ARGS", agent_path).split() if os.environ.get("MCP_SERVER_ARGS") else [agent_path]
MCP_TOOLS_CACHE_TTL = int(os.environ.get("MCP_TOOLS_CACHE_TTL", "60"))

# Global model storage
global_medical_models = {}
global_medical_tokenizers = {}
global_file_info = {}
global_tts_model = None
global_embed_model = None

# MCP client storage
global_mcp_session = None
global_mcp_stdio_ctx = None
global_mcp_lock = threading.Lock()
global_mcp_tools_cache = {"timestamp": 0.0, "tools": None}

# UI constants
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

