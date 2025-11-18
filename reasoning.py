"""Autonomous reasoning and execution planning"""
import json
import asyncio
from logger import logger
from client import MCP_AVAILABLE, call_agent
from config import GEMINI_MODEL

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None


async def autonomous_reasoning_gemini(query: str) -> dict:
    """Autonomous reasoning using Gemini MCP"""
    reasoning_prompt = f"""Analyze this medical query and provide structured reasoning:
Query: "{query}"
Analyze:
1. Query Type: (diagnosis, treatment, drug_info, symptom_analysis, research, general_info)
2. Complexity: (simple, moderate, complex, multi_faceted)
3. Information Needs: What specific information is required?
4. Requires RAG: (yes/no) - Does this need document context?
5. Requires Web Search: (yes/no) - Does this need current/updated information?
6. Sub-questions: Break down into key sub-questions if complex
Respond in JSON format:
{{
    "query_type": "...",
    "complexity": "...",
    "information_needs": ["..."],
    "requires_rag": true/false,
    "requires_web_search": true/false,
    "sub_questions": ["..."]
}}"""
    
    system_prompt = "You are a medical reasoning system. Analyze queries systematically and provide structured JSON responses."
    
    response = await call_agent(
        user_prompt=reasoning_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            reasoning = json.loads(response[json_start:json_end])
        else:
            raise ValueError("No JSON found")
    except:
        reasoning = {
            "query_type": "general_info",
            "complexity": "moderate",
            "information_needs": ["medical information"],
            "requires_rag": True,
            "requires_web_search": False,
            "sub_questions": [query]
        }
    
    logger.info(f"Reasoning analysis: {reasoning}")
    return reasoning


def autonomous_reasoning(query: str, history: list) -> dict:
    """Autonomous reasoning: Analyze query complexity, intent, and information needs"""
    if not MCP_AVAILABLE:
        logger.info("ℹ️ Gemini MCP not available for reasoning, using fallback (app will continue to work normally)")
        return {
            "query_type": "general_info",
            "complexity": "moderate",
            "information_needs": ["medical information"],
            "requires_rag": True,
            "requires_web_search": False,
            "sub_questions": [query]
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                reasoning = nest_asyncio.run(autonomous_reasoning_gemini(query))
                return reasoning
            else:
                logger.error("Error in nested async reasoning: nest_asyncio not available")
        else:
            reasoning = loop.run_until_complete(autonomous_reasoning_gemini(query))
            return reasoning
    except Exception as e:
        logger.error(f"Gemini MCP reasoning error: {e}")
    
    logger.warning("⚠️ Falling back to default reasoning")
    return {
        "query_type": "general_info",
        "complexity": "moderate",
        "information_needs": ["medical information"],
        "requires_rag": True,
        "requires_web_search": False,
        "sub_questions": [query]
    }


def create_execution_plan(reasoning: dict, query: str, has_rag_index: bool) -> dict:
    """Planning: Create multi-step execution plan based on reasoning analysis"""
    plan = {
        "steps": [],
        "strategy": "sequential",
        "iterations": 1
    }
    
    if reasoning["complexity"] in ["complex", "multi_faceted"]:
        plan["strategy"] = "iterative"
        plan["iterations"] = 2
    
    plan["steps"].append({
        "step": 1,
        "action": "detect_language",
        "description": "Detect query language and translate if needed"
    })
    
    if reasoning.get("requires_rag", True) and has_rag_index:
        plan["steps"].append({
            "step": 2,
            "action": "rag_retrieval",
            "description": "Retrieve relevant document context",
            "parameters": {"top_k": 15, "merge_threshold": 0.5}
        })
    
    if reasoning.get("requires_web_search", False):
        plan["steps"].append({
            "step": 3,
            "action": "web_search",
            "description": "Search web for current/updated information",
            "parameters": {"max_results": 5}
        })
    
    if reasoning.get("sub_questions") and len(reasoning["sub_questions"]) > 1:
        plan["steps"].append({
            "step": 4,
            "action": "multi_step_reasoning",
            "description": "Process sub-questions iteratively",
            "sub_questions": reasoning["sub_questions"]
        })
    
    plan["steps"].append({
        "step": len(plan["steps"]) + 1,
        "action": "synthesize_answer",
        "description": "Generate comprehensive answer from all sources"
    })
    
    if reasoning["complexity"] in ["complex", "multi_faceted"]:
        plan["steps"].append({
            "step": len(plan["steps"]) + 1,
            "action": "self_reflection",
            "description": "Evaluate answer quality and completeness"
        })
    
    logger.info(f"Execution plan created: {len(plan['steps'])} steps")
    return plan


def autonomous_execution_strategy(reasoning: dict, plan: dict, use_rag: bool, use_web_search: bool, has_rag_index: bool) -> dict:
    """Autonomous execution: Make decisions on information gathering strategy"""
    strategy = {
        "use_rag": use_rag,
        "use_web_search": use_web_search,
        "reasoning_override": False,
        "rationale": ""
    }
    
    if reasoning.get("requires_web_search", False) and not use_web_search:
        strategy["rationale"] = "Reasoning suggests web search for current information, but the user kept it disabled."
    
    if strategy["rationale"]:
        logger.info(f"Autonomous reasoning note: {strategy['rationale']}")
    
    return strategy

