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
- Automatic summarization of web search results using DeepSeek-R1
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
- Powered by DeepSeek-R1-8B for translation

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
- **Translation Model**: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
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
   - Retrieves relevant patient history from documents
   - Searches web for latest ADA/ADA-EASD guidelines
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

### **Implementation in Clinical Settings**

**Hospital Systems**: Deploy for clinical decision support, integrating with EMR systems and institutional medical libraries.

**Specialty Clinics**: Customize for specific medical specialties by uploading specialty-specific documents and guidelines.

**Medical Education**: Support medical training and education with comprehensive, evidence-based answers.

**Research Institutions**: Accelerate medical research by synthesizing information from multiple sources.

---

**Note**: This system is designed to **assist** medical professionals with information retrieval and synthesis. It does not replace clinical judgment. All medical decisions should be made by qualified healthcare professionals who consider the full clinical context, patient-specific factors, and their professional expertise.

> Introduction: A medical app for MCP-1st-Birthday hackathon, integrating MCP searcher and document RAG with autonomous reasoning, planning, and execution capabilities for enterprise-level clinical decision support.