---
title: MedLLM Agent
emoji: 🩺
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: 'MedicalMCP RAG & Search with MedSwin'
tags:
  - mcp-in-action-track-enterprise
---

# 🩺 MedLLM Agent

**Advanced Medical AI Assistant** powered by fine-tuned MedSwin models with comprehensive knowledge retrieval capabilities.

## ✨ Key Features

### 📄 **Document RAG (Retrieval-Augmented Generation)**
- Upload medical documents (PDF/TXT) and get answers based on your uploaded content
- Hierarchical document indexing with auto-merging retrieval
- Mitigates hallucination by grounding responses in your documents
- Toggle RAG on/off - when disabled, provides concise clinical answers without document context

### 🌐 **Web Search Integration (MCP Protocol)**
- Fetch knowledge from reliable online medical resources
- Automatic summarization of web search results using Llama-8B
- Enriches context for medical specialist models
- Combines document RAG + web sources for comprehensive answers

### 🧠 **MedSwin Medical Specialist Models**
- **MedSwin SFT** (default) - Supervised Fine-Tuned model
- **MedSwin KD** - Knowledge Distillation model  
- **MedSwin TA** - Task-Aware merged model
- Models download on-demand for efficient resource usage
- Fine-tuned on MedAlpaca-7B for medical domain expertise

### 🌍 **Multi-Language Support**
- Automatic language detection
- Non-English queries automatically translated to English
- Medical model processes in English
- Responses translated back to original language
- Powered by Llama-3.1-8B-Instruct for translation

### ⚙️ **Advanced Configuration**
- Customizable generation parameters (temperature, top-p, top-k)
- Adjustable retrieval settings (top-k, merge threshold)
- Increased max tokens to prevent early stopping
- Custom EOS handling for medical models
- Dynamic system prompts based on RAG status

## 🚀 Usage

1. **Upload Documents**: Drag and drop PDF or text files containing medical information
2. **Configure Settings**: 
   - Enable/disable Document RAG
   - Enable/disable Web Search (MCP)
   - Select medical model (MedSwin SFT/KD/TA)
3. **Ask Questions**: Type your medical question in any language
4. **Get Answers**: Receive comprehensive answers based on:
   - Your uploaded documents (if RAG enabled)
   - Web sources (if web search enabled)
   - Medical model's training knowledge

## 🔧 Technical Details

- **Medical Models**: MedSwin/MedSwin-7B-SFT, MedSwin-7B-KD, MedSwin-Merged-TA-SFT-0.7
- **Translation Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Embedding Model**: abhinand/MedEmbed-large-v0.1 (domain-tuned medical embeddings)
- **RAG Framework**: LlamaIndex with hierarchical node parsing
- **Web Search**: DuckDuckGo with content extraction and summarization

## 📋 Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- transformers, torch
- llama-index
- langdetect
- duckduckgo-search
- gradio, spaces

## 🎯 Use Cases

- Medical document Q&A
- Clinical information retrieval
- Medical research assistance
- Multi-language medical consultations
- Evidence-based medical answers

---

**Note**: This system is designed to assist with medical information retrieval. Always consult qualified healthcare professionals for medical decisions.

> Introduction: A medical app for MCP-1st-Birthday hackathon, integrate MCP searcher and document RAG