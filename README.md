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
  - mcp-in-action-track-creative
  - building-mcp-track-enterprise
  - building-mcp-track-creative
---

[Demo](https://huggingface.co/spaces/MCP-1st-Birthday/MedLLM-Agent)  

# 🩺 MedLLM Agent

**Advanced Medical AI Assistant** powered by fine-tuned MedSwin models with comprehensive knowledge retrieval capabilities.

## ✨ Key Features

### 📄 **Document RAG (Retrieval-Augmented Generation)**
- Upload medical documents (PDF, Word, TXT, MD, JSON, XML, CSV) and get answers based on your uploaded content
- Document parsing powered by Gemini MCP for accurate text extraction
- Hierarchical document indexing with auto-merging retrieval
- Mitigates hallucination by grounding responses in your documents
- Toggle RAG on/off - when disabled, provides concise clinical answers without document context

### 🌐 **Web Search Integration (MCP Protocol)**
- **Native MCP Support**: Uses Model Context Protocol (MCP) tools for web search and content extraction
- **Automatic Fallback**: Gracefully falls back to direct library calls if MCP is not configured
- **Configurable MCP Servers**: Connect to any MCP-compatible search server via environment variables
- **Content Extraction**: Automatically fetches and extracts full content from search results using MCP tools
- **Automatic Summarization**: Summarizes web search results using Gemini MCP
- **Enriches Context**: Combines document RAG + web sources for comprehensive answers

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
- Powered by Gemini MCP for translation

### 🎤 **Voice Features**
- **Speech-to-Text**: Microphone icon for voice input transcription using Gemini MCP
- **Text-to-Speech**: Speaker icon in responses to generate voice output using Maya1 TTS model
- Speech-to-text powered by Gemini MCP for accurate transcription

### ⚙️ **Advanced Configuration**
- Customizable generation parameters (temperature, top-p, top-k)
- Adjustable retrieval settings (top-k, merge threshold)
- Increased max tokens to prevent early stopping
- Custom EOS handling for medical models
- Dynamic system prompts based on RAG status

## 🚀 Usage

1. **Upload Documents**: Drag and drop PDF, Word, or text files containing medical information
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
- **Translation**: Gemini MCP (gemini-2.5-flash-lite for simple tasks)
- **Document Parsing**: Gemini MCP (supports PDF, Word, TXT, MD, JSON, XML, CSV)
- **Speech-to-Text**: Gemini MCP (gemini-2.5-flash-lite)
- **Summarization**: Gemini MCP (gemini-2.5-flash for complex tasks)
- **Reasoning & Reflection**: Gemini MCP (gemini-2.5-flash)
- **Text-to-Speech**: maya-research/maya1
- **Embedding Model**: abhinand/MedEmbed-large-v0.1 (domain-tuned medical embeddings)
- **RAG Framework**: LlamaIndex with hierarchical node parsing
- **Web Search**: Model Context Protocol (MCP) tools with automatic fallback to DuckDuckGo
- **MCP Client**: Python MCP SDK for standardized tool integration
- **Gemini MCP Server**: aistudio-mcp-server via MCP protocol

## 📋 Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- **MCP Integration**: `mcp`, `nest-asyncio` (primary - for MCP protocol support)
- **Fallback Dependencies**: `requests`, `beautifulsoup4` (used when MCP is not available)
- **Core ML**: `transformers`, `torch`
- **RAG Framework**: `llama-index`
- **Utilities**: `langdetect`, `gradio`, `spaces`

### 🔌 MCP Configuration

The application uses Gemini MCP (Model Context Protocol) for translation, document parsing, transcription, and summarization. Configure Gemini MCP server via environment variables:

```bash
# Gemini MCP Server (required)
export GEMINI_API_KEY="your-gemini-api-key"

# Gemini MCP Server Configuration
export MCP_SERVER_COMMAND="python"
export MCP_SERVER_ARGS="-m aistudio_mcp_server"

# Optional Gemini Configuration
export GEMINI_MODEL="gemini-2.5-flash"  # For harder tasks (default)
export GEMINI_MODEL_LITE="gemini-2.5-flash-lite"  # For parsing and simple tasks (default)
export GEMINI_TIMEOUT=300000  # Request timeout in milliseconds (default: 5 minutes)
export GEMINI_MAX_OUTPUT_TOKENS=8192  # Maximum output tokens (default)
export GEMINI_MAX_FILES=10  # Maximum number of files per request (default)
export GEMINI_MAX_TOTAL_FILE_SIZE=50  # Maximum total file size in MB (default)
export GEMINI_TEMPERATURE=0.2  # Temperature for generation 0-2 (default: 0.2)
```

**Available Gemini MCP Tools:**
- **Translation**: Multi-language translation using Gemini MCP (gemini-2.5-flash-lite)
- **Document Parsing**: Extract text from PDF, Word, and other documents using Gemini MCP
- **Speech-to-Text**: Audio transcription using Gemini MCP (gemini-2.5-flash-lite)
- **Summarization**: Web content summarization using Gemini MCP (gemini-2.5-flash)
- **Reasoning & Reflection**: Query analysis and answer quality evaluation using Gemini MCP

**Supported File Types for Document Parsing:**
- Documents: PDF, DOC, DOCX (treated as images, one page = one image)
- Text: TXT, MD, JSON, XML, CSV
- Images: JPG, JPEG, PNG, GIF, WebP, SVG, BMP, TIFF
- Audio: MP3, WAV, AIFF, AAC, OGG, FLAC (up to 15MB per file)
- Video: MP4, AVI, MOV, WEBM, FLV, MPG, WMV (up to 10 files per request)

1. **Install MCP Python SDK** (already in requirements.txt):
   ```bash
   pip install mcp nest-asyncio
   ```

2. **Install Gemini MCP Server**:
   ```bash
   # Install Python package
   pip install aistudio-mcp-server
   ```

3. **Get Gemini API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/) to get your API key
   - Set it as an environment variable: `export GEMINI_API_KEY="your-api-key"`

4. **Configure via Environment Variables**:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export MCP_SERVER_COMMAND="python"
   export MCP_SERVER_ARGS="-m aistudio_mcp_server"
   ```

**Note**: The application requires Gemini MCP for translation, document parsing, transcription, and summarization. Web search functionality still supports fallback to direct library calls if MCP is not configured.

## 🎯 Use Cases

- Medical document Q&A
- Clinical information retrieval
- Medical research assistance
- Multi-language medical consultations
- Evidence-based medical answers

## 🏥 Enterprise-Level Clinical Decision Support

### **Empowering Medical Specialists with AI-Powered Decision Support**

MedLLM Agent is designed to support **doctors, clinicians, and medical specialists** in making informed clinical decisions by leveraging the power of Large Language Models (LLMs) and Model Context Protocol (MCP). This system transforms how medical professionals access, analyze, and synthesize medical information in real-time.

### **Key Enterprise Capabilities**

#### 1. **Autonomous Reasoning & Planning**
- **Intelligent Query Analysis**: The system autonomously analyzes medical queries to understand:
  - Query type (diagnosis, treatment, drug information, symptom analysis)
  - Complexity level (simple, moderate, complex, multi-faceted)
  - Information requirements and data sources needed
  
- **Multi-Step Execution Planning**: For complex clinical questions, the system:
  - Breaks down queries into sub-questions
  - Creates structured execution plans
  - Determines optimal information gathering strategies
  - Adapts approach based on query complexity

#### 2. **Autonomous Decision-Making**
- **Smart Resource Selection**: The system autonomously decides:
  - When to use document RAG vs. web search
  - When both sources are needed for comprehensive answers
  - Optimal retrieval parameters based on query characteristics
  
- **Context-Aware Execution**: Automatically:
  - Overrides user settings when reasoning suggests better approaches
  - Combines multiple information sources intelligently
  - Prioritizes evidence-based medical sources

#### 3. **Self-Reflection & Quality Assurance**
- **Answer Quality Evaluation**: For complex queries, the system:
  - Self-evaluates answer completeness and accuracy
  - Identifies missing information or aspects
  - Provides improvement suggestions
  - Ensures high-quality clinical responses

### **Enterprise Use Cases for Medical Specialists**

#### **Clinical Decision Support**
- **Diagnostic Assistance**: Upload patient records, lab results, and medical histories. Ask complex diagnostic questions and receive evidence-based answers grounded in your documents and current medical literature.

- **Treatment Planning**: Query treatment protocols, drug interactions, and therapeutic guidelines. The system autonomously retrieves relevant information from your clinical documents and current medical databases.

- **Drug Information & Interactions**: Get comprehensive drug information, contraindications, and interaction analyses by combining your formulary documents with up-to-date web sources.

#### **Research & Evidence Synthesis**
- **Literature Review Support**: Upload research papers, clinical trials, and medical literature. The system helps synthesize findings, identify connections, and answer research questions.

- **Clinical Guideline Analysis**: Compare and analyze multiple clinical guidelines, protocols, and best practices from your document library.

#### **Multi-Language Clinical Support**
- **International Patient Care**: Handle queries in multiple languages. The system automatically translates, processes with medical models, and translates responses back—enabling care for diverse patient populations.

#### **Real-Time Information Access**
- **Current Medical Knowledge**: Leverage MCP web search to access:
  - Latest treatment protocols
  - Recent clinical trial results
  - Updated drug information
  - Current medical guidelines
- **MCP Protocol Benefits**: Standardized, modular tool integration allows easy switching between search providers and enhanced reliability

### **How It Works: Autonomous Reasoning in Action**

1. **Query Analysis** → System analyzes: "What are the treatment options for Type 2 diabetes in elderly patients with renal impairment?"
   - Identifies as complex, multi-faceted query
   - Determines need for both RAG (patient records) and web search (current guidelines)
   - Breaks into sub-questions: treatment options, age considerations, renal function impact

2. **Autonomous Planning** → Creates execution plan:
   - Step 1: Language detection/translation
   - Step 2: RAG retrieval from patient documents
   - Step 3: Web search for current diabetes treatment guidelines
   - Step 4: Multi-step reasoning for each sub-question
   - Step 5: Synthesis of comprehensive answer
   - Step 6: Self-reflection on answer quality

3. **Autonomous Execution** → System executes plan:
   - Retrieves relevant patient history from documents (parsed via Gemini MCP)
   - Searches web for latest ADA/ADA-EASD guidelines using MCP tools
   - Fetches and extracts full content from search results via MCP
   - Summarizes web content using Gemini MCP
   - Synthesizes information considering age and renal function
   - Generates evidence-based treatment recommendations

4. **Self-Reflection** → Evaluates answer:
   - Checks completeness (all sub-questions addressed?)
   - Verifies accuracy (evidence-based?)
   - Suggests improvements if needed

### **Enterprise Benefits**

✅ **Time Efficiency**: Reduces time spent searching through documents and medical databases  
✅ **Evidence-Based Decisions**: Grounds answers in uploaded documents and current medical literature  
✅ **Reduced Hallucination**: RAG ensures answers are based on actual documents and verified sources  
✅ **Comprehensive Coverage**: Combines institutional knowledge (documents) with current medical knowledge (web)  
✅ **Quality Assurance**: Self-reflection ensures high-quality, complete answers  
✅ **Scalability**: Handles multiple languages, complex queries, and large document libraries  
✅ **Clinical Workflow Integration**: Designed to fit into existing clinical decision-making processes  
✅ **MCP Protocol**: Standardized tool integration for reliable, maintainable web search capabilities

### **Implementation in Clinical Settings**

**Hospital Systems**: Deploy for clinical decision support, integrating with EMR systems and institutional medical libraries.

**Specialty Clinics**: Customize for specific medical specialties by uploading specialty-specific documents and guidelines.

**Medical Education**: Support medical training and education with comprehensive, evidence-based answers.

**Research Institutions**: Accelerate medical research by synthesizing information from multiple sources.

---

**Note**: This system is designed to **assist** medical professionals with information retrieval and synthesis. It does not replace clinical judgment. All medical decisions should be made by qualified healthcare professionals who consider the full clinical context, patient-specific factors, and their professional expertise.

> Introduction: A medical app for MCP-1st-Birthday hackathon, integrating MCP searcher and document RAG with autonomous reasoning, planning, and execution capabilities for enterprise-level clinical decision support.