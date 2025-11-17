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
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Gemini imports
try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed. Install with: pip install google-genai", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set in environment variables")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

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
    return [
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
            
            # Prepare model
            try:
                gemini_model = genai.GenerativeModel(model)
            except Exception as e:
                logger.error(f"Error loading model {model}: {e}")
                return [TextContent(type="text", text=f"Error: Failed to load model {model}")]
            
            # Prepare content parts
            parts = []
            
            # Add system instruction if provided
            if system_prompt:
                # Gemini models use system_instruction parameter
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS
                )
            else:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS
                )
            
            # Prepare content parts for Gemini
            # Gemini API accepts a list where each part can be:
            # - A string (for text)
            # - A dict with "mime_type" and "data" keys (for binary data)
            content_parts = []
            
            # Prepare files if provided
            if files:
                gemini_files = prepare_gemini_files(files)
                for file_part in gemini_files:
                    # Use genai.types.Part or dict format
                    content_parts.append({
                        "mime_type": file_part["mime_type"],
                        "data": file_part["data"]
                    })
            
            # Add text prompt (as string)
            content_parts.append(user_prompt)
            
            # Generate content
            try:
                if system_prompt:
                    # Use system_instruction for models that support it
                    response = await asyncio.to_thread(
                        gemini_model.generate_content,
                        content_parts,
                        generation_config=generation_config,
                        system_instruction=system_prompt
                    )
                else:
                    response = await asyncio.to_thread(
                        gemini_model.generate_content,
                        content_parts,
                        generation_config=generation_config
                    )
                
                # Extract text from response
                if response and response.text:
                    return [TextContent(type="text", text=response.text)]
                else:
                    return [TextContent(type="text", text="Error: No response from Gemini")]
                    
            except Exception as e:
                logger.error(f"Error generating content: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
                
        except Exception as e:
            logger.error(f"Error in generate_content: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main entry point"""
    logger.info("Starting Gemini MCP Server...")
    logger.info(f"Gemini API Key: {'Set' if GEMINI_API_KEY else 'Not Set'}")
    logger.info(f"Default Model: {GEMINI_MODEL}")
    logger.info(f"Default Lite Model: {GEMINI_MODEL_LITE}")
    
    # Use stdio_server from mcp.server.stdio
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as streams:
        await app.run(
            streams[0],  # read_stream
            streams[1],  # write_stream
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())

