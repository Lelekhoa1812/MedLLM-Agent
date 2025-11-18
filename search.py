"""Web search functions"""
import json
import asyncio
import concurrent.futures
from logger import logger
from client import MCP_AVAILABLE, get_mcp_session, get_cached_mcp_tools, call_agent
from config import GEMINI_MODEL

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None


async def search_web_mcp_tool(query: str, max_results: int = 5) -> list:
    """Search web using MCP web search tool (e.g., DuckDuckGo MCP server)"""
    if not MCP_AVAILABLE:
        return []
    
    try:
        tools = await get_cached_mcp_tools()
        if not tools:
            return []
        
        search_tool = None
        for tool in tools:
            tool_name_lower = tool.name.lower()
            if any(keyword in tool_name_lower for keyword in ["search", "duckduckgo", "ddg", "web"]):
                search_tool = tool
                logger.info(f"Found web search MCP tool: {tool.name}")
                break
        
        if not search_tool:
            tools = await get_cached_mcp_tools(force_refresh=True)
            for tool in tools:
                tool_name_lower = tool.name.lower()
                if any(keyword in tool_name_lower for keyword in ["search", "duckduckgo", "ddg", "web"]):
                    search_tool = tool
                    logger.info(f"Found web search MCP tool after refresh: {tool.name}")
                    break
        
        if search_tool:
            try:
                session = await get_mcp_session()
                if session is None:
                    return []
                
                result = await session.call_tool(
                    search_tool.name,
                    arguments={"query": query, "max_results": max_results}
                )
            
                web_content = []
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            try:
                                data = json.loads(item.text)
                                if isinstance(data, list):
                                    for entry in data[:max_results]:
                                        web_content.append({
                                            'title': entry.get('title', ''),
                                            'url': entry.get('url', entry.get('href', '')),
                                            'content': entry.get('body', entry.get('snippet', entry.get('content', '')))
                                        })
                                elif isinstance(data, dict):
                                    if 'results' in data:
                                        for entry in data['results'][:max_results]:
                                            web_content.append({
                                                'title': entry.get('title', ''),
                                                'url': entry.get('url', entry.get('href', '')),
                                                'content': entry.get('body', entry.get('snippet', entry.get('content', '')))
                                            })
                                    else:
                                        web_content.append({
                                            'title': data.get('title', ''),
                                            'url': data.get('url', data.get('href', '')),
                                            'content': data.get('body', data.get('snippet', data.get('content', '')))
                                        })
                            except json.JSONDecodeError:
                                web_content.append({
                                    'title': '',
                                    'url': '',
                                    'content': item.text[:1000]
                                })
                
                if web_content:
                    return web_content
            except Exception as e:
                logger.error(f"Error calling web search MCP tool: {e}")
        
        else:
            logger.debug("No MCP web search tool discovered in current catalog")
            return []
    except Exception as e:
        logger.error(f"Web search MCP tool error: {e}")
        return []


async def search_web_mcp(query: str, max_results: int = 5) -> list:
    """Search web using MCP tools - tries web search MCP tool first, then falls back to direct search"""
    results = await search_web_mcp_tool(query, max_results)
    if results:
        logger.info(f"✅ Web search via MCP tool: found {len(results)} results")
        return results
    
    logger.info("ℹ️ [Direct API] No web search MCP tool found, using direct DuckDuckGo search (results will be summarized with Gemini MCP)")
    return search_web_fallback(query, max_results)


def search_web_fallback(query: str, max_results: int = 5) -> list:
    """Fallback web search using DuckDuckGo directly (when MCP is not available)"""
    logger.info(f"🔍 [Direct API] Performing web search using DuckDuckGo API for: {query[:100]}...")
    try:
        from ddgs import DDGS
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("Fallback dependencies (ddgs, requests, beautifulsoup4) not available")
        return []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            web_content = []
            for result in results:
                try:
                    url = result.get('href', '')
                    title = result.get('title', '')
                    snippet = result.get('body', '')
                    
                    try:
                        response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'html.parser')
                            for script in soup(["script", "style"]):
                                script.decompose()
                            text = soup.get_text()
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)
                            if len(text) > 1000:
                                text = text[:1000] + "..."
                            web_content.append({
                                'title': title,
                                'url': url,
                                'content': snippet + "\n" + text[:500] if text else snippet
                            })
                        else:
                            web_content.append({
                                'title': title,
                                'url': url,
                                'content': snippet
                            })
                    except:
                        web_content.append({
                            'title': title,
                            'url': url,
                            'content': snippet
                        })
                except Exception as e:
                    logger.error(f"Error processing search result: {e}")
                    continue
            logger.info(f"✅ [Direct API] Web search completed: {len(web_content)} results")
            return web_content
    except Exception as e:
        logger.error(f"❌ [Direct API] Web search error: {e}")
        return []


def search_web(query: str, max_results: int = 5) -> list:
    """Search web using MCP tools (synchronous wrapper) - prioritizes MCP over direct ddgs"""
    if MCP_AVAILABLE:
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                if nest_asyncio:
                    results = nest_asyncio.run(search_web_mcp(query, max_results))
                    if results:
                        return results
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, search_web_mcp(query, max_results))
                        results = future.result(timeout=30)
                        if results:
                            return results
            else:
                results = loop.run_until_complete(search_web_mcp(query, max_results))
                if results:
                    return results
        except Exception as e:
            logger.error(f"Error running async MCP search: {e}")
    
    logger.info("ℹ️ [Direct API] Falling back to direct DuckDuckGo search (MCP unavailable or returned no results)")
    return search_web_fallback(query, max_results)


async def summarize_web_content_gemini(content_list: list, query: str) -> str:
    """Summarize web search results using Gemini MCP"""
    combined_content = "\n\n".join([f"Source: {item['title']}\n{item['content']}" for item in content_list[:3]])
    
    user_prompt = f"""Summarize the following web search results related to the query: "{query}"
Extract key medical information, facts, and insights. Be concise and focus on reliable information.
Search Results:
{combined_content}
Summary:"""
    
    system_prompt = "You are a medical information summarizer. Extract and summarize key medical facts accurately."
    
    result = await call_agent(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.5
    )
    
    return result.strip()


def summarize_web_content(content_list: list, query: str) -> str:
    """Summarize web search results using Gemini MCP"""
    if not MCP_AVAILABLE:
        logger.warning("Gemini MCP not available for summarization")
        if content_list:
            return content_list[0].get('content', '')[:500]
        return ""
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                summary = nest_asyncio.run(summarize_web_content_gemini(content_list, query))
                if summary:
                    return summary
            else:
                logger.error("Error in nested async summarization: nest_asyncio not available")
        else:
            summary = loop.run_until_complete(summarize_web_content_gemini(content_list, query))
            if summary:
                return summary
    except Exception as e:
        logger.error(f"Gemini MCP summarization error: {e}")
    
    if content_list:
        return content_list[0].get('content', '')[:500]
    return ""

