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
- Hierarchical document indexing with auto-merging retrieval for comprehensive context
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
- **MedSwin TA** (default) - Task-Aware merged model
- **MedSwin SFT** - Supervised Fine-Tuned model
- **MedSwin KD** - Knowledge Distillation model  
- Models download on-demand for efficient resource usage
- Fine-tuned on MedAlpaca-7B for medical domain expertise

### 🌍 **Multi-Language Support**
- Automatic language detection
- Non-English queries automatically translated to English
- Medical model processes in English
- Responses translated back to original language
- Powered by Gemini MCP for translation

### 🎤 **Voice Features**
- **Speech-to-Text**: Voice input transcription using Gemini MCP
- **Inline Mic Experience**: Built-in microphone widget with live recording timer that drops transcripts straight into the chat box
- **Text-to-Speech**: Voice output generation using Maya1 TTS model (optional, fallback to MCP if unavailable) plus a one-click "Play Response" control for the latest answer

### 🤝 **MAC Architecture (Multi-Agent Collaboration)**
- **Gemini Supervisor**: Orchestrates query processing by breaking queries into flexible sub-topics (up to 10 based on complexity, explores different approaches)
- **MedSwin Specialist**: Executes tasks sequentially, providing concise clinical answers
- **Enhanced Synthesis**: Supervisor synthesizes all MedSwin responses with clear context into comprehensive final answers
- **Iterative Improvement**: Supervisor challenges and enhances answers until confirmed optimal (up to 2 iterations)
- **Search Mode**: Gemini creates 1-4 search strategies → executes ddgs searches (1-2 sources each) → summarizes briefly
- **Conditional Search Trigger**: When search mode enabled, supervisor can trigger additional searches if answer is unclear or has gaps
- **RAG Mode**: Gemini brainstorms retrieved documents into 1-4 short contexts for MedSwin decision-making
- **Clean Output**: All internal thoughts/conversations are logged only; users see only the final answer
- **Markdown Format**: Final answers use bullet points (tables automatically converted)
- **Deterministic Mode**: `Disable agentic reasoning` switch runs MedSwin alone for offline-friendly, model-only answers

### ⚙️ **Advanced Configuration**
- Customizable generation parameters (temperature, top-p, top-k)
- Adjustable retrieval settings (top-k, merge threshold)
- Increased max tokens to prevent early stopping
- Custom EOS handling for medical models
- Dynamic system prompts based on RAG status
- One-click agentic toggle to run MedSwin alone (no RAG/web search) for deterministic, offline-safe answers

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
- **Architecture**: MAC (Multi-Agent Collaboration) - Gemini Supervisor + MedSwin Specialist
- **Translation**: Gemini MCP (gemini-2.5-flash-lite)
- **Document Parsing**: Gemini MCP (PDF, Word, TXT, MD, JSON, XML, CSV)
- **Speech-to-Text**: Gemini MCP (gemini-2.5-flash-lite)
- **Supervisor Tasks**: Gemini MCP (gemini-2.5-flash) - query breakdown, search strategies, RAG brainstorming
- **MedSwin Execution**: GPU-tagged tasks for efficient inference
- **Text-to-Speech**: maya-research/maya1 (optional, with MCP fallback)
- **Embedding Model**: abhinand/MedEmbed-large-v0.1 (domain-tuned medical embeddings)
- **RAG Framework**: LlamaIndex with hierarchical node parsing and auto-merging retrieval
- **Web Search**: MCP tools with automatic fallback to DuckDuckGo
- **MCP Server**: Bundled Python-based Gemini MCP server (agent.py)

## 📋 Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- **MCP Integration**: `mcp`, `nest-asyncio`, `google-genai` (for Gemini MCP server)
- **Fallback Dependencies**: `requests`, `beautifulsoup4`, `ddgs` (used when MCP web search unavailable)
- **Core ML**: `transformers`, `torch`, `accelerate`
- **RAG Framework**: `llama-index`, `llama_index.llms.huggingface`, `llama_index.embeddings.huggingface`
- **Utilities**: `langdetect`, `gradio`, `spaces`, `soundfile`
- **TTS**: Optional - `TTS` package (voice features work with MCP fallback if unavailable)

### 🔌 MCP Configuration

The application uses a bundled Gemini MCP server (agent.py) for translation, document parsing, transcription, and summarization. Configure via environment variables

**Setup Steps:**

1. **Install Dependencies** (already in requirements.txt):
   ```bash
   pip install mcp nest-asyncio google-genai
   ```

2. **Get Gemini API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/) to get your API key
   - Set it: `export GEMINI_API_KEY="your-api-key"`

3. **Run the Application**:
   - The bundled MCP server (agent.py) will be used automatically
   - No additional MCP server installation required

**Note**: The application requires Gemini MCP for translation, document parsing, transcription, and summarization. Web search supports fallback to direct DuckDuckGo API if MCP web search tools are unavailable.

## 🎯 Use Cases

- **Clinical Decision Support**: Evidence-based answers from documents and current medical literature
- **Medical Document Q&A**: Query uploaded patient records, research papers, and clinical guidelines
- **Multi-Language Consultations**: Automatic translation for international patient care
- **Research Assistance**: Synthesize information from multiple medical sources
- **Drug Information**: Comprehensive drug information with interaction analysis

## 🏥 Enterprise-Level Clinical Decision Support

### **Empowering Medical Specialists with AI-Powered Decision Support**

MedLLM Agent is designed to support **doctors, clinicians, and medical specialists** in making informed clinical decisions by leveraging the power of Large Language Models (LLMs) and Model Context Protocol (MCP). This system transforms how medical professionals access, analyze, and synthesize medical information in real-time.

### **Key Enterprise Capabilities**

#### 1. **MAC Architecture (Multi-Agent Collaboration)**
- **Gemini Supervisor Agent**: 
  - Breaks user queries into flexible sub-topics (up to 10 based on complexity, explores different approaches/angles)
  - Synthesizes all MedSwin responses with clear context into comprehensive final answers
  - Challenges and enhances answers iteratively until confirmed optimal (up to 2 iterations)
  - In search mode: creates 1-4 search strategies, executes ddgs (1-2 sources each), summarizes briefly
  - Conditional search trigger: Can trigger additional searches if answer is unclear or has gaps (only when search mode enabled)
  - In RAG mode: brainstorms retrieved documents into 1-4 concise contexts
  - All supervisor decisions logged internally, not displayed
  
- **MedSwin Specialist Agent**:
  - Executes tasks assigned by Gemini Supervisor (GPU-tagged)
  - Processes each sub-topic sequentially with focused context
  - Generates concise, clinically accurate answers
  - Returns Markdown format with bullet points (tables auto-converted)

#### 2. **Clean User Experience**
- **Internal Thoughts Hidden**: All Gemini-MedSwin conversations logged only
- **Final Answer Only**: Users see only the polished, final answer
- **Structured Output**: Markdown bullets, no internal planning tables
- **Efficient Processing**: Contexts kept brief to respect token limits

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

### **How It Works: MAC Architecture in Action**

1. **Gemini Supervisor - Query Breakdown** → Analyzes query and breaks into flexible sub-topics (up to 10 based on complexity):
   - Example: "What are the treatment options for Type 2 diabetes in elderly patients with renal impairment?"
   - Explores different approaches (clinical, diagnostic, treatment, prevention perspectives)
   - Creates structured sub-topics: treatment options, age considerations, renal function impact, drug interactions, monitoring protocols
   - Number of subtasks adapts to query complexity (not limited to 4)
   - All planning logged internally, not displayed to user

2. **Gemini Supervisor - Context Preparation**:
   - **Search Mode**: Creates 1-4 search strategies → executes ddgs (1-2 sources each) → summarizes briefly
   - **RAG Mode**: Retrieves documents → brainstorms into 1-4 concise contexts for MedSwin
   - Contexts kept brief to respect MedSwin token limits

3. **MedSwin Specialist - Task Execution** (GPU-tagged):
   - Executes each sub-topic task sequentially
   - Receives focused context from Gemini Supervisor
   - Generates concise clinical answers (Markdown bullets, no tables)
   - All execution logged internally

4. **Gemini Supervisor - Answer Synthesis**:
   - Synthesizes all MedSwin responses with clear context
   - Integrates information from all sub-topics seamlessly
   - Creates coherent, comprehensive final answer
   - Provides better context than simple concatenation

5. **Gemini Supervisor - Challenge & Enhancement Loop**:
   - Evaluates answer quality (completeness, accuracy, clarity)
   - Challenges answer if not optimal
   - Provides specific enhancement instructions
   - Enhances answer iteratively (up to 2 iterations)
   - Continues until answer confirmed optimal

6. **Conditional Search Trigger** (only when search mode enabled):
   - Supervisor checks if answer is unclear or has gaps
   - If needed, generates specific search queries to fill gaps
   - Executes additional searches
   - Enhances answer with new search context

7. **Final Answer Assembly**:
   - Converts any tables to Markdown bullets
   - Adds citations if web sources used
   - Translates back if needed
   - **Only final answer displayed** - all internal thoughts remain in logs

### **Enterprise Benefits**

✅ **Time Efficiency**: Reduces time spent searching through documents and medical databases  
✅ **Evidence-Based Decisions**: Grounds answers in uploaded documents and current medical literature  
✅ **Reduced Hallucination**: RAG ensures answers are based on actual documents and verified sources  
✅ **Comprehensive Coverage**: Combines institutional knowledge (documents) with current medical knowledge (web)  
✅ **Enhanced Quality**: Iterative challenge loop ensures answers are optimal before delivery  
✅ **Flexible Task Breakdown**: Adapts to query complexity with flexible subtask generation (not limited to 4 steps)  
✅ **Intelligent Search**: Conditional search trigger fills gaps when answers are unclear  
✅ **Better Context**: Enhanced synthesis provides clearer, more comprehensive final answers  
✅ **Scalability**: Handles multiple languages, complex queries, and large document libraries  
✅ **Clinical Workflow Integration**: Designed to fit into existing clinical decision-making processes  
✅ **MCP Protocol**: Standardized tool integration for reliable, maintainable web search capabilities

### **Implementation in Clinical Settings**

- **Hospital Systems**: Clinical decision support with EMR integration and institutional medical libraries
- **Specialty Clinics**: Customize with specialty-specific documents and guidelines
- **Medical Education**: Comprehensive, evidence-based answers for training and education
- **Research Institutions**: Accelerate research by synthesizing information from multiple sources

---

**⚠️ Important Disclaimer**: This system is designed to **assist** medical professionals with information retrieval and synthesis. It does not replace clinical judgment. All medical decisions must be made by qualified healthcare professionals who consider the full clinical context, patient-specific factors, and their professional expertise.

---

> **Built for MCP-1st-Birthday Hackathon**: Enterprise-level clinical decision support system integrating MCP protocol, document RAG, and autonomous reasoning capabilities.