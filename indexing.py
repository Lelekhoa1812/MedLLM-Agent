"""Document parsing and indexing functions"""
import os
import base64
import asyncio
import tempfile
import time
import gradio as gr
import spaces
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Document as LlamaDocument,
)
from llama_index.core import Settings
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from tqdm import tqdm
from logger import logger
from mcp import MCP_AVAILABLE, call_agent
import config
from models import get_llm_for_rag, get_or_create_embed_model

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None


async def parse_document_gemini(file_path: str, file_extension: str) -> str:
    """Parse document using Gemini MCP"""
    if not MCP_AVAILABLE:
        return ""
    
    try:
        with open(file_path, 'rb') as f:
            file_content = base64.b64encode(f.read()).decode('utf-8')
        
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.csv': 'text/csv'
        }
        mime_type = mime_type_map.get(file_extension, 'application/octet-stream')
        
        files = [{
            "content": file_content,
            "type": mime_type
        }]
        
        system_prompt = "Extract all text content from the document accurately."
        user_prompt = "Extract all text content from this document. Return only the extracted text, preserving structure and formatting where possible."
        
        result = await call_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            files=files,
            model=config.GEMINI_MODEL_LITE,
            temperature=0.2
        )
        
        return result.strip()
    except Exception as e:
        logger.error(f"Gemini document parsing error: {e}")
        return ""


def extract_text_from_document(file):
    """Extract text from document using Gemini MCP"""
    file_name = file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension == '.txt':
        text = file.read().decode('utf-8')
        return text, len(text.split()), None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            file.seek(0)
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        if MCP_AVAILABLE:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    if nest_asyncio:
                        text = nest_asyncio.run(parse_document_gemini(tmp_file_path, file_extension))
                    else:
                        logger.error("Error in nested async document parsing: nest_asyncio not available")
                        text = ""
                else:
                    text = loop.run_until_complete(parse_document_gemini(tmp_file_path, file_extension))
                
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                
                if text:
                    return text, len(text.split()), None
                else:
                    return None, 0, ValueError(f"Failed to extract text from {file_extension} file using Gemini MCP")
            except Exception as e:
                logger.error(f"Gemini MCP document parsing error: {e}")
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                return None, 0, ValueError(f"Error parsing {file_extension} file: {str(e)}")
        else:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            return None, 0, ValueError(f"Gemini MCP not available. Cannot parse {file_extension} files.")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return None, 0, ValueError(f"Error processing {file_extension} file: {str(e)}")


@spaces.GPU(max_duration=120)
def create_or_update_index(files, request: gr.Request):
    """Create or update RAG index from uploaded files"""
    if not files:
        return "Please provide files.", ""
    
    start_time = time.time()
    user_id = request.session_hash
    save_dir = f"./{user_id}_index"
    
    llm = get_llm_for_rag()
    embed_model = get_or_create_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model
    file_stats = []
    new_documents = []
    
    for file in tqdm(files, desc="Processing files"):
        file_basename = os.path.basename(file.name)
        text, word_count, error = extract_text_from_document(file)
        if error:
            logger.error(f"Error processing file {file_basename}: {str(error)}")
            file_stats.append({
                "name": file_basename,
                "words": 0,
                "status": f"error: {str(error)}"
            })
            continue
        
        doc = LlamaDocument(
            text=text,
            metadata={
                "file_name": file_basename,
                "word_count": word_count,
                "source": "user_upload"
            }
        )
        new_documents.append(doc)
        
        file_stats.append({
            "name": file_basename,
            "words": word_count,
            "status": "processed"
        })
        
        config.global_file_info[file_basename] = {
            "word_count": word_count,
            "processed_at": time.time()
        }
    
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128],
        chunk_overlap=20
    )
    logger.info(f"Parsing {len(new_documents)} documents into hierarchical nodes")
    new_nodes = node_parser.get_nodes_from_documents(new_documents)
    new_leaf_nodes = get_leaf_nodes(new_nodes)
    new_root_nodes = get_root_nodes(new_nodes)
    logger.info(f"Generated {len(new_nodes)} total nodes ({len(new_root_nodes)} root, {len(new_leaf_nodes)} leaf)")
    
    if os.path.exists(save_dir):
        logger.info(f"Loading existing index from {save_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=save_dir)
        index = load_index_from_storage(storage_context, settings=Settings)
        docstore = storage_context.docstore
        
        docstore.add_documents(new_nodes)
        for node in tqdm(new_leaf_nodes, desc="Adding leaf nodes to index"):
            index.insert_nodes([node])
            
        total_docs = len(docstore.docs)
        logger.info(f"Updated index with {len(new_nodes)} new nodes from {len(new_documents)} files")
    else:
        logger.info("Creating new index")
        docstore = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(docstore=docstore)
        docstore.add_documents(new_nodes)
        
        index = VectorStoreIndex(
            new_leaf_nodes,
            storage_context=storage_context,
            settings=Settings
        )
        total_docs = len(new_documents)
        logger.info(f"Created new index with {len(new_nodes)} nodes from {len(new_documents)} files")
    
    index.storage_context.persist(persist_dir=save_dir)
    
    file_list_html = "<div class='file-list'>"
    for stat in file_stats:
        status_color = "#4CAF50" if stat["status"] == "processed" else "#f44336"
        file_list_html += f"<div><span style='color:{status_color}'>●</span> {stat['name']} - {stat['words']} words</div>"
    file_list_html += "</div>"
    processing_time = time.time() - start_time
    stats_output = f"<div class='stats-box'>"
    stats_output += f"✓ Processed {len(files)} files in {processing_time:.2f} seconds<br>"
    stats_output += f"✓ Created {len(new_nodes)} nodes ({len(new_leaf_nodes)} leaf nodes)<br>"
    stats_output += f"✓ Total documents in index: {total_docs}<br>"
    stats_output += f"✓ Index saved to: {save_dir}<br>"
    stats_output += "</div>"
    output_container = f"<div class='info-container'>"
    output_container += file_list_html
    output_container += stats_output
    output_container += "</div>"
    return f"Successfully indexed {len(files)} files.", output_container

