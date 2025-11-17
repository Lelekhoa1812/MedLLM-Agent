#!/usr/bin/env python3
"""
Gemini MCP Server
A Python-based MCP server that provides Gemini AI capabilities via Model Context Protocol.
This server implements the generate_content tool for translation, summarization, document parsing, and transcription.
"""

import os
import sys
import json
import base64
import asyncio
import logging
from typing import Any, Sequence
from pathlib import Path

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server.models import InitializationOptions
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Gemini imports
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Error: google-genai not installed. Install with: pip install google-genai", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MCP logging to WARNING to reduce noise
mcp_logger = logging.getLogger("mcp")
mcp_logger.setLevel(logging.WARNING)
root_logger = logging.getLogger("root")
root_logger.setLevel(logging.INFO)

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set in environment variables")
    sys.exit(1)

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Configuration from environment
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_LITE = os.environ.get("GEMINI_MODEL_LITE", "gemini-2.5-flash-lite")
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "300000"))  # milliseconds
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
GEMINI_MAX_FILES = int(os.environ.get("GEMINI_MAX_FILES", "10"))
GEMINI_MAX_TOTAL_FILE_SIZE = int(os.environ.get("GEMINI_MAX_TOTAL_FILE_SIZE", "50"))  # MB
GEMINI_TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))

# Create MCP server
app = Server("gemini-mcp-server")

def decode_base64_file(content: str, mime_type: str = None) -> bytes:
    """Decode base64 encoded file content"""
    try:
        return base64.b64decode(content)
    except Exception as e:
        logger.error(f"Error decoding base64 content: {e}")
        raise

def prepare_gemini_files(files: list) -> list:
    """Prepare files for Gemini API"""
    gemini_parts = []
    
    for file_obj in files:
        try:
            # Handle file with path
            if "path" in file_obj:
                file_path = file_obj["path"]
                mime_type = file_obj.get("type")
                
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Read file
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                # Auto-detect MIME type if not provided
                if not mime_type:
                    from mimetypes import guess_type
                    mime_type, _ = guess_type(file_path)
                    if not mime_type:
                        mime_type = "application/octet-stream"
            
            # Handle file with base64 content
            elif "content" in file_obj:
                file_data = decode_base64_file(file_obj["content"])
                mime_type = file_obj.get("type", "application/octet-stream")
            else:
                logger.warning("File object must have either 'path' or 'content'")
                continue
            
            # Add to Gemini parts
            gemini_parts.append({
                "mime_type": mime_type,
                "data": file_data
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            continue
    
    return gemini_parts

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    try:
        tools = [
            Tool(
                name="generate_content",
                description="Generate content using Gemini AI. Supports text generation, translation, summarization, document parsing, and audio transcription.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_prompt": {
                            "type": "string",
                            "description": "User prompt for generation (required)"
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "System prompt to guide AI behavior (optional)"
                        },
                        "files": {
                            "type": "array",
                            "description": "Array of files to include in generation (optional)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string", "description": "Path to file"},
                                    "content": {"type": "string", "description": "Base64 encoded file content"},
                                    "type": {"type": "string", "description": "MIME type (auto-detected from file extension)"}
                                }
                            }
                        },
                        "model": {
                            "type": "string",
                            "description": f"Gemini model to use (default: {GEMINI_MODEL})"
                        },
                        "temperature": {
                            "type": "number",
                            "description": f"Temperature for generation 0-2 (default: {GEMINI_TEMPERATURE})"
                        }
                    },
                    "required": ["user_prompt"]
                }
            )
        ]
        return tools
    except Exception as e:
        logger.error(f"Error in list_tools(): {e}")
        raise

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls"""
    if name == "generate_content":
        try:
            user_prompt = arguments.get("user_prompt")
            if not user_prompt:
                return [TextContent(type="text", text="Error: user_prompt is required")]
            
            system_prompt = arguments.get("system_prompt")
            files = arguments.get("files", [])
            model = arguments.get("model", GEMINI_MODEL)
            temperature = float(arguments.get("temperature", GEMINI_TEMPERATURE))
            
            # Prepare content for Gemini API
            # The API accepts contents as a string or list
            # For files, we need to handle them differently
            contents = user_prompt
            
            # If system prompt is provided, prepend it to the user prompt
            if system_prompt:
                contents = f"{system_prompt}\n\n{user_prompt}"
            
            # Prepare content for Gemini API
            # The google-genai API expects contents as a list of parts
            gemini_contents = []
            
            # Add text content as first part
            gemini_contents.append(contents)
            
            # Add file content if provided
            if files:
                try:
                    file_parts = prepare_gemini_files(files)
                    # Convert file parts to the format expected by Gemini API
                    for file_part in file_parts:
                        # The API expects parts with inline_data for binary content
                        gemini_contents.append({
                            "inline_data": {
                                "mime_type": file_part["mime_type"],
                                "data": base64.b64encode(file_part["data"]).decode('utf-8')
                            }
                        })
                    logger.info(f"Added {len(file_parts)} file(s) to Gemini request")
                except Exception as e:
                    logger.warning(f"Error preparing files: {e}, continuing with text only")
            
            # Generate content using Gemini API
            try:
                # Get the model instance
                gemini_model = gemini_client.models.get(model)
                
                # Prepare generation config
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS
                }
                
                # Use asyncio.to_thread to make the blocking call async
                # The API accepts contents as a list and config as a separate parameter
                def generate_sync():
                    return gemini_model.generate_content(
                        contents=gemini_contents,
                        config=generation_config
                    )
                
                response = await asyncio.to_thread(generate_sync)
                
                # Extract text from response
                if response and hasattr(response, 'text') and response.text:
                    return [TextContent(type="text", text=response.text)]
                elif response and hasattr(response, 'candidates') and response.candidates:
                    # Try to extract text from candidates if response is a list of candidates
                    text_parts = []
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    text_parts.append(part.text)
                    if text_parts:
                        text = ''.join(text_parts)
                        return [TextContent(type="text", text=text)]
                    else:
                        logger.warning("Gemini returned response but no text found")
                        return [TextContent(type="text", text="Error: No text in Gemini response")]
                else:
                    logger.warning("Gemini returned empty response")
                    return [TextContent(type="text", text="Error: No response from Gemini")]
                    
            except Exception as e:
                logger.error(f"Error generating content: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
                
        except Exception as e:
            logger.error(f"Error in generate_content: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Starting Gemini MCP Server...")
    logger.info(f"Gemini API Key: {'Set' if GEMINI_API_KEY else 'Not Set'}")
    logger.info(f"Default Model: {GEMINI_MODEL}")
    logger.info(f"Default Lite Model: {GEMINI_MODEL_LITE}")
    logger.info("=" * 60)
    
    # Use stdio_server from mcp.server.stdio
    from mcp.server.stdio import stdio_server
    
    # Keep logging enabled for debugging
    original_root_level = logging.getLogger("root").level
    logging.getLogger("root").setLevel(logging.INFO)
    
    try:
        async with stdio_server() as streams:
            # Prepare server capabilities for initialization
            try:
                if hasattr(app, "get_capabilities"):
                    server_capabilities = app.get_capabilities()
                else:
                    server_capabilities = {}
            except Exception as cap_error:
                logger.warning(f"Failed to gather server capabilities: {cap_error}")
                server_capabilities = {}

            init_options = InitializationOptions(
                server_name="gemini-mcp-server",
                server_version="1.0.0",
                capabilities=server_capabilities,
            )

            logger.info("MCP server ready")
            try:
                # Run the server - it will automatically handle the initialization handshake
                await app.run(
                    read_stream=streams[0],
                    write_stream=streams[1],
                    initialization_options=init_options,
                )
            except Exception as run_error:
                logger.error(f"Error in app.run(): {run_error}")
                raise
    except Exception as e:
        logging.getLogger("root").setLevel(original_root_level)
        logger.error(f"MCP server fatal error: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())