"""MCP session management and tool caching"""
import os
import time
import asyncio
from logger import logger
import config

# MCP imports
MCP_CLIENT_INFO = None
MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp import types as mcp_types
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    MCP_CLIENT_INFO = mcp_types.Implementation(
        name="MedLLM-Agent",
        version=os.environ.get("SPACE_VERSION", "local"),
    )
    logger.info("✅ MCP SDK imported successfully")
except ImportError as e:
    logger.warning(f"❌ MCP SDK import failed: {e}")
    logger.info("   Install with: pip install mcp>=0.1.0")
    logger.info("   The app will continue to work with fallback functionality (direct API calls)")
    MCP_AVAILABLE = False
    MCP_CLIENT_INFO = None
except Exception as e:
    logger.error(f"❌ Unexpected error initializing MCP: {type(e).__name__}: {e}")
    logger.info("   The app will continue to work with fallback functionality")
    MCP_AVAILABLE = False
    MCP_CLIENT_INFO = None


async def get_mcp_session():
    """Get or create MCP client session with proper context management"""
    if not MCP_AVAILABLE:
        logger.warning("MCP not available - SDK not installed")
        return None
    
    if config.global_mcp_session is not None:
        return config.global_mcp_session
    
    try:
        mcp_env = os.environ.copy()
        if config.GEMINI_API_KEY:
            mcp_env["GEMINI_API_KEY"] = config.GEMINI_API_KEY
            logger.info(f"✅ GEMINI_API_KEY found: {config.GEMINI_API_KEY[:10]}...{config.GEMINI_API_KEY[-4:]}")
        else:
            logger.warning("❌ GEMINI_API_KEY not set in environment. Gemini MCP features will not work.")
            logger.warning("   Set it with: export GEMINI_API_KEY='your-api-key'")
            return None
        
        if os.environ.get("GEMINI_MODEL"):
            mcp_env["GEMINI_MODEL"] = os.environ.get("GEMINI_MODEL")
        if os.environ.get("GEMINI_TIMEOUT"):
            mcp_env["GEMINI_TIMEOUT"] = os.environ.get("GEMINI_TIMEOUT")
        if os.environ.get("GEMINI_MAX_OUTPUT_TOKENS"):
            mcp_env["GEMINI_MAX_OUTPUT_TOKENS"] = os.environ.get("GEMINI_MAX_OUTPUT_TOKENS")
        if os.environ.get("GEMINI_TEMPERATURE"):
            mcp_env["GEMINI_TEMPERATURE"] = os.environ.get("GEMINI_TEMPERATURE")
        
        logger.info(f"Creating MCP client session... (command: {config.MCP_SERVER_COMMAND} {config.MCP_SERVER_ARGS})")
        
        server_params = StdioServerParameters(
            command=config.MCP_SERVER_COMMAND,
            args=config.MCP_SERVER_ARGS,
            env=mcp_env
        )
        
        stdio_ctx = stdio_client(server_params)
        read, write = await stdio_ctx.__aenter__()
        
        session = ClientSession(
            read,
            write,
            client_info=MCP_CLIENT_INFO,
        )
        
        try:
            await session.__aenter__()
            init_result = await session.initialize()
            server_info = getattr(init_result, "serverInfo", None)
            server_name = getattr(server_info, "name", "unknown")
            server_version = getattr(server_info, "version", "unknown")
            logger.info(f"✅ MCP session initialized successfully (server={server_name} v{server_version})")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"❌ MCP session initialization failed: {error_type}: {error_msg}")
            logger.error(f"   This might be due to:")
            logger.error(f"   - Invalid GEMINI_API_KEY")
            logger.error(f"   - agent.py server not starting correctly")
            logger.error(f"   - Network/firewall issues")
            import traceback
            logger.debug(f"   Full traceback: {traceback.format_exc()}")
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            return None
        
        config.global_mcp_session = session
        config.global_mcp_stdio_ctx = stdio_ctx
        logger.info("✅ MCP client session created successfully")
        return session
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"❌ Failed to create MCP client session: {error_type}: {error_msg}")
        config.global_mcp_session = None
        config.global_mcp_stdio_ctx = None
        return None


def invalidate_mcp_tools_cache():
    """Invalidate cached MCP tool metadata"""
    config.global_mcp_tools_cache = {"timestamp": 0.0, "tools": None}


async def get_cached_mcp_tools(force_refresh: bool = False):
    """Return cached MCP tools list to avoid repeated list_tools calls"""
    if not MCP_AVAILABLE:
        return []
    
    now = time.time()
    if (
        not force_refresh
        and config.global_mcp_tools_cache["tools"]
        and now - config.global_mcp_tools_cache["timestamp"] < config.MCP_TOOLS_CACHE_TTL
    ):
        return config.global_mcp_tools_cache["tools"]
    
    session = await get_mcp_session()
    if session is None:
        return []
    
    try:
        tools_resp = await session.list_tools()
        tools_list = list(getattr(tools_resp, "tools", []) or [])
        config.global_mcp_tools_cache = {"timestamp": now, "tools": tools_list}
        return tools_list
    except Exception as e:
        logger.error(f"Failed to refresh MCP tools: {e}")
        invalidate_mcp_tools_cache()
        return []


async def test_mcp_connection() -> bool:
    """Test MCP connection and return True if successful"""
    if not MCP_AVAILABLE:
        logger.warning("Cannot test MCP: SDK not available")
        return False
    
    if not config.GEMINI_API_KEY:
        logger.warning("Cannot test MCP: GEMINI_API_KEY not set")
        return False
    
    try:
        session = await get_mcp_session()
        if session is None:
            logger.warning("MCP connection test failed: Could not create session")
            return False
        
        # Try to list tools as a connectivity test
        tools = await get_cached_mcp_tools()
        if tools:
            logger.info(f"✅ MCP connection test successful! Found {len(tools)} tools")
            return True
        else:
            logger.warning("MCP connection test: Session created but no tools found")
            return False
    except Exception as e:
        logger.error(f"MCP connection test failed: {type(e).__name__}: {e}")
        return False


async def call_agent(user_prompt: str, system_prompt: str = None, files: list = None, model: str = None, temperature: float = 0.2) -> str:
    """Call Gemini MCP generate_content tool"""
    if not MCP_AVAILABLE:
        logger.debug("MCP not available for Gemini call")
        return ""
    
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set - cannot use Gemini MCP")
        return ""
    
    try:
        session = await get_mcp_session()
        if session is None:
            logger.warning("Failed to get MCP session for Gemini call - check GEMINI_API_KEY and agent.py")
            # Invalidate session to force retry on next call
            config.global_mcp_session = None
            return ""
        
        tools = await get_cached_mcp_tools()
        if not tools:
            tools = await get_cached_mcp_tools(force_refresh=True)
        if not tools:
            logger.error("Unable to obtain MCP tool catalog for Gemini calls")
            return ""
        
        generate_tool = None
        for tool in tools:
            if tool.name == "generate_content" or "generate_content" in tool.name.lower():
                generate_tool = tool
                logger.info(f"Found Gemini MCP tool: {tool.name}")
                break
        
        if not generate_tool:
            logger.warning(f"Gemini MCP generate_content tool not found. Available tools: {[t.name for t in tools]}")
            invalidate_mcp_tools_cache()
            return ""
        
        arguments = {
            "user_prompt": user_prompt
        }
        if system_prompt:
            arguments["system_prompt"] = system_prompt
        if files:
            arguments["files"] = files
        if model:
            arguments["model"] = model
        if temperature is not None:
            arguments["temperature"] = temperature
        
        result = await session.call_tool(generate_tool.name, arguments=arguments)
        
        if hasattr(result, 'content') and result.content:
            for item in result.content:
                if hasattr(item, 'text'):
                    response_text = item.text.strip()
                    return response_text
        logger.warning("⚠️ Gemini MCP returned empty or invalid result")
        return ""
    except Exception as e:
        logger.error(f"Gemini MCP call error: {e}")
        return ""

