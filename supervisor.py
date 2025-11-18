"""Gemini Supervisor functions for MAC architecture"""
import json
import asyncio
import torch
import spaces
from logger import logger
from client import MCP_AVAILABLE, call_agent
from config import GEMINI_MODEL, GEMINI_MODEL_LITE
from utils import format_prompt_manually

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None


async def gemini_supervisor_breakdown_async(query: str, use_rag: bool, use_web_search: bool, time_elapsed: float, max_duration: int = 120) -> dict:
    """Gemini Supervisor: Break user query into sub-topics"""
    remaining_time = max(15, max_duration - time_elapsed)
    
    mode_description = []
    if use_rag:
        mode_description.append("RAG mode enabled - will use retrieved documents")
    if use_web_search:
        mode_description.append("Web search mode enabled - will search online sources")
    if not mode_description:
        mode_description.append("Direct answer mode - no additional context")
    
    estimated_time_per_task = 8
    max_topics_by_time = max(2, int((remaining_time - 20) / estimated_time_per_task))
    max_topics = min(max_topics_by_time, 10)
    
    prompt = f"""You are a supervisor agent coordinating with a MedSwin medical specialist model.
Break the following medical query into focused sub-topics that MedSwin can answer sequentially.
Explore different potential approaches to comprehensively address the topic.

Query: "{query}"
Mode: {', '.join(mode_description)}
Time Remaining: ~{remaining_time:.1f}s
Maximum Topics: {max_topics} (adjust based on complexity - use as many as needed for thorough coverage)

Return ONLY valid JSON (no markdown, no tables, no explanations):
{{
  "sub_topics": [
    {{
      "id": 1,
      "topic": "concise topic name",
      "instruction": "specific directive for MedSwin to answer this topic",
      "expected_tokens": 200,
      "priority": "high|medium|low",
      "approach": "brief description of approach/angle for this topic"
    }},
    ...
  ],
  "strategy": "brief strategy description explaining the breakdown approach",
  "exploration_note": "brief note on different approaches explored"
}}

Guidelines:
- Break down the query into as many subtasks as needed for comprehensive coverage
- Explore different angles/approaches (e.g., clinical, diagnostic, treatment, prevention, research perspectives)
- Each topic should be focused and answerable in ~200 tokens by MedSwin
- Prioritize topics by importance (high priority first)
- Don't limit yourself to 4 topics - use more if the query is complex or multi-faceted"""
    
    system_prompt = "You are a medical query supervisor. Break queries into structured JSON sub-topics, exploring different approaches. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            breakdown = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Query broken into {len(breakdown.get('sub_topics', []))} sub-topics")
            return breakdown
        else:
            raise ValueError("Supervisor JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Breakdown parsing failed: {exc}")
        breakdown = {
            "sub_topics": [
                {"id": 1, "topic": "Core Question", "instruction": "Address the main medical question", "expected_tokens": 200, "priority": "high", "approach": "direct answer"},
                {"id": 2, "topic": "Clinical Details", "instruction": "Provide key clinical insights", "expected_tokens": 200, "priority": "medium", "approach": "clinical perspective"},
            ],
            "strategy": "Sequential answer with key points",
            "exploration_note": "Fallback breakdown - basic coverage"
        }
        logger.warning(f"[GEMINI SUPERVISOR] Using fallback breakdown")
        return breakdown


async def gemini_supervisor_search_strategies_async(query: str, time_elapsed: float) -> dict:
    """Gemini Supervisor: In search mode, break query into 1-4 searching strategies"""
    prompt = f"""You are supervising web search for a medical query.
Break this query into 1-4 focused search strategies (each targeting 1-2 sources).

Query: "{query}"

Return ONLY valid JSON:
{{
  "search_strategies": [
    {{
      "id": 1,
      "strategy": "search query string",
      "target_sources": 1,
      "focus": "what to search for"
    }},
    ...
  ],
  "max_strategies": 4
}}

Keep strategies focused and avoid overlap."""
    
    system_prompt = "You are a search strategy supervisor. Create focused search queries. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,
        temperature=0.2
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            strategies = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Created {len(strategies.get('search_strategies', []))} search strategies")
            return strategies
        else:
            raise ValueError("Search strategies JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Search strategies parsing failed: {exc}")
        return {
            "search_strategies": [
                {"id": 1, "strategy": query, "target_sources": 2, "focus": "main query"}
            ],
            "max_strategies": 1
        }


async def gemini_supervisor_rag_brainstorm_async(query: str, retrieved_docs: str, time_elapsed: float) -> dict:
    """Gemini Supervisor: In RAG mode, brainstorm retrieved documents into 1-4 short contexts"""
    max_doc_length = 3000
    if len(retrieved_docs) > max_doc_length:
        retrieved_docs = retrieved_docs[:max_doc_length] + "..."
    
    prompt = f"""You are supervising RAG context preparation for a medical query.
Brainstorm the retrieved documents into 1-4 concise, focused contexts that MedSwin can use.

Query: "{query}"
Retrieved Documents:
{retrieved_docs}

Return ONLY valid JSON:
{{
  "contexts": [
    {{
      "id": 1,
      "context": "concise summary of relevant information (keep under 500 chars)",
      "focus": "what this context covers",
      "relevance": "high|medium|low"
    }},
    ...
  ],
  "max_contexts": 4
}}

Keep contexts brief and factual. Avoid redundancy."""
    
    system_prompt = "You are a RAG context supervisor. Summarize documents into concise contexts. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,
        temperature=0.2
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            contexts = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Brainstormed {len(contexts.get('contexts', []))} RAG contexts")
            return contexts
        else:
            raise ValueError("RAG contexts JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] RAG brainstorming parsing failed: {exc}")
        return {
            "contexts": [
                {"id": 1, "context": retrieved_docs[:500], "focus": "retrieved information", "relevance": "high"}
            ],
            "max_contexts": 1
        }


def gemini_supervisor_breakdown(query: str, use_rag: bool, use_web_search: bool, time_elapsed: float, max_duration: int = 120) -> dict:
    """Wrapper to obtain supervisor breakdown synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP SDK unavailable, using fallback breakdown")
        return {
            "sub_topics": [
                {"id": 1, "topic": "Core Question", "instruction": "Address the main medical question", "expected_tokens": 200, "priority": "high", "approach": "direct answer"},
                {"id": 2, "topic": "Clinical Details", "instruction": "Provide key clinical insights", "expected_tokens": 200, "priority": "medium", "approach": "clinical perspective"},
            ],
            "strategy": "Sequential answer with key points",
            "exploration_note": "Fallback breakdown - basic coverage"
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                try:
                    return nest_asyncio.run(
                        gemini_supervisor_breakdown_async(query, use_rag, use_web_search, time_elapsed, max_duration)
                    )
                except Exception as e:
                    logger.error(f"[GEMINI SUPERVISOR] Async breakdown failed: {e}")
                    raise
            else:
                logger.error("[GEMINI SUPERVISOR] Nested breakdown execution failed: nest_asyncio not available")
                raise RuntimeError("nest_asyncio not available")
        return loop.run_until_complete(
            gemini_supervisor_breakdown_async(query, use_rag, use_web_search, time_elapsed, max_duration)
        )
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Breakdown request failed: {type(exc).__name__}: {exc}")
        logger.warning("[GEMINI SUPERVISOR] Falling back to default breakdown")
        return {
            "sub_topics": [
                {"id": 1, "topic": "Core Question", "instruction": "Address the main medical question", "expected_tokens": 200, "priority": "high", "approach": "direct answer"},
            ],
            "strategy": "Direct answer",
            "exploration_note": "Fallback breakdown - single topic"
        }


def gemini_supervisor_search_strategies(query: str, time_elapsed: float) -> dict:
    """Wrapper to obtain search strategies synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP unavailable for search strategies")
        return {
            "search_strategies": [
                {"id": 1, "strategy": query, "target_sources": 2, "focus": "main query"}
            ],
            "max_strategies": 1
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                return nest_asyncio.run(gemini_supervisor_search_strategies_async(query, time_elapsed))
            else:
                logger.error("[GEMINI SUPERVISOR] Nested search strategies execution failed: nest_asyncio not available")
        return loop.run_until_complete(gemini_supervisor_search_strategies_async(query, time_elapsed))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Search strategies request failed: {exc}")
        return {
            "search_strategies": [
                {"id": 1, "strategy": query, "target_sources": 2, "focus": "main query"}
            ],
            "max_strategies": 1
        }


def gemini_supervisor_rag_brainstorm(query: str, retrieved_docs: str, time_elapsed: float) -> dict:
    """Wrapper to obtain RAG brainstorm synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP unavailable for RAG brainstorm")
        return {
            "contexts": [
                {"id": 1, "context": retrieved_docs[:500], "focus": "retrieved information", "relevance": "high"}
            ],
            "max_contexts": 1
        }
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                return nest_asyncio.run(gemini_supervisor_rag_brainstorm_async(query, retrieved_docs, time_elapsed))
            else:
                logger.error("[GEMINI SUPERVISOR] Nested RAG brainstorm execution failed: nest_asyncio not available")
        return loop.run_until_complete(gemini_supervisor_rag_brainstorm_async(query, retrieved_docs, time_elapsed))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] RAG brainstorm request failed: {exc}")
        return {
            "contexts": [
                {"id": 1, "context": retrieved_docs[:500], "focus": "retrieved information", "relevance": "high"}
            ],
            "max_contexts": 1
        }


@spaces.GPU(max_duration=120)
def execute_medswin_task(
    medical_model_obj,
    medical_tokenizer,
    task_instruction: str,
    context: str,
    system_prompt_base: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    penalty: float
) -> str:
    """MedSwin Specialist: Execute a single task assigned by Gemini Supervisor"""
    if context:
        full_prompt = f"{system_prompt_base}\n\nContext:\n{context}\n\nTask: {task_instruction}\n\nAnswer concisely with key bullet points (Markdown format, no tables):"
    else:
        full_prompt = f"{system_prompt_base}\n\nTask: {task_instruction}\n\nAnswer concisely with key bullet points (Markdown format, no tables):"
    
    messages = [{"role": "system", "content": full_prompt}]
    
    if hasattr(medical_tokenizer, 'chat_template') and medical_tokenizer.chat_template is not None:
        try:
            prompt = medical_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"[MEDSWIN] Chat template failed, using manual formatting: {e}")
            prompt = format_prompt_manually(messages, medical_tokenizer)
    else:
        prompt = format_prompt_manually(messages, medical_tokenizer)
    
    inputs = medical_tokenizer(prompt, return_tensors="pt").to(medical_model_obj.device)
    
    eos_token_id = medical_tokenizer.eos_token_id or medical_tokenizer.pad_token_id
    
    with torch.no_grad():
        outputs = medical_model_obj.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, 800),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=penalty,
            do_sample=True,
            eos_token_id=eos_token_id,
            pad_token_id=medical_tokenizer.pad_token_id or eos_token_id
        )
    
    response = medical_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    response = response.strip()
    if "|" in response and "---" in response:
        logger.warning("[MEDSWIN] Detected table format, converting to Markdown bullets")
        lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('|') and '---' not in line]
        response = '\n'.join([f"- {line}" if not line.startswith('-') else line for line in lines])
    
    logger.info(f"[MEDSWIN] Task completed: {len(response)} chars generated")
    return response


async def gemini_supervisor_synthesize_async(query: str, medswin_answers: list, rag_contexts: list, search_contexts: list, breakdown: dict) -> str:
    """Gemini Supervisor: Synthesize final answer from all MedSwin responses"""
    context_summary = ""
    if rag_contexts:
        context_summary += f"Document Context Available: {len(rag_contexts)} context(s) from uploaded documents.\n"
    if search_contexts:
        context_summary += f"Web Search Context Available: {len(search_contexts)} search result(s).\n"
    
    all_answers_text = "\n\n---\n\n".join([f"## {i+1}. {ans}" for i, ans in enumerate(medswin_answers)])
    
    prompt = f"""You are a supervisor agent synthesizing a comprehensive medical answer from multiple specialist responses.

Original Query: "{query}"

Context Available:
{context_summary}

MedSwin Specialist Responses (from {len(medswin_answers)} sub-topics):
{all_answers_text}

Your task:
1. Synthesize all responses into a coherent, comprehensive final answer
2. Integrate information from all sub-topics seamlessly
3. Ensure the answer directly addresses the original query
4. Maintain clinical accuracy and clarity
5. Use clear structure with appropriate headings and bullet points
6. Remove redundancy and contradictions
7. Ensure all important points from MedSwin responses are included

Return the final synthesized answer in Markdown format. Do not add meta-commentary or explanations - just provide the final answer."""
    
    system_prompt = "You are a medical answer synthesis supervisor. Create comprehensive, well-structured final answers from multiple specialist responses."
    
    result = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    return result.strip()


async def gemini_supervisor_challenge_async(query: str, current_answer: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> dict:
    """Gemini Supervisor: Challenge and evaluate the current answer"""
    context_info = ""
    if rag_contexts:
        context_info += f"Document contexts: {len(rag_contexts)} available.\n"
    if search_contexts:
        context_info += f"Search contexts: {len(search_contexts)} available.\n"
    
    all_answers_text = "\n\n---\n\n".join([f"## {i+1}. {ans}" for i, ans in enumerate(medswin_answers)])
    
    prompt = f"""You are a supervisor agent evaluating and challenging a medical answer for quality and completeness.

Original Query: "{query}"

Available Context:
{context_info}

MedSwin Specialist Responses:
{all_answers_text}

Current Synthesized Answer:
{current_answer[:2000]}

Evaluate this answer and provide:
1. Completeness: Does it fully address the query? What's missing?
2. Accuracy: Are there any inaccuracies or contradictions?
3. Clarity: Is it well-structured and clear?
4. Context Usage: Are document/search contexts properly utilized?
5. Improvement Suggestions: Specific ways to enhance the answer

Return ONLY valid JSON:
{{
  "is_optimal": true/false,
  "completeness_score": 0-10,
  "accuracy_score": 0-10,
  "clarity_score": 0-10,
  "missing_aspects": ["..."],
  "inaccuracies": ["..."],
  "improvement_suggestions": ["..."],
  "needs_more_context": true/false,
  "enhancement_instructions": "specific instructions for improving the answer"
}}"""
    
    system_prompt = "You are a medical answer quality evaluator. Provide honest, constructive feedback in JSON format. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            evaluation = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Challenge evaluation: optimal={evaluation.get('is_optimal', False)}, scores={evaluation.get('completeness_score', 'N/A')}/{evaluation.get('accuracy_score', 'N/A')}/{evaluation.get('clarity_score', 'N/A')}")
            return evaluation
        else:
            raise ValueError("Evaluation JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Challenge evaluation parsing failed: {exc}")
        return {
            "is_optimal": True,
            "completeness_score": 7,
            "accuracy_score": 7,
            "clarity_score": 7,
            "missing_aspects": [],
            "inaccuracies": [],
            "improvement_suggestions": [],
            "needs_more_context": False,
            "enhancement_instructions": ""
        }


async def gemini_supervisor_enhance_answer_async(query: str, current_answer: str, enhancement_instructions: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> str:
    """Gemini Supervisor: Enhance the answer based on challenge feedback"""
    context_info = ""
    if rag_contexts:
        context_info += f"Document contexts: {len(rag_contexts)} available.\n"
    if search_contexts:
        context_info += f"Search contexts: {len(search_contexts)} available.\n"
    
    all_answers_text = "\n\n---\n\n".join([f"## {i+1}. {ans}" for i, ans in enumerate(medswin_answers)])
    
    prompt = f"""You are a supervisor agent enhancing a medical answer based on evaluation feedback.

Original Query: "{query}"

Available Context:
{context_info}

MedSwin Specialist Responses:
{all_answers_text}

Current Answer (to enhance):
{current_answer}

Enhancement Instructions:
{enhancement_instructions}

Create an enhanced version of the answer that:
1. Addresses all improvement suggestions
2. Fills in missing aspects
3. Corrects any inaccuracies
4. Improves clarity and structure
5. Better utilizes available context
6. Maintains all valuable information from the current answer

Return the enhanced answer in Markdown format. Do not add meta-commentary."""
    
    system_prompt = "You are a medical answer enhancement supervisor. Improve answers based on evaluation feedback while maintaining accuracy."
    
    result = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    return result.strip()


async def gemini_supervisor_check_clarity_async(query: str, answer: str, use_web_search: bool) -> dict:
    """Gemini Supervisor: Check if answer is unclear or supervisor is unsure"""
    if not use_web_search:
        return {"is_unclear": False, "needs_search": False, "search_queries": []}
    
    prompt = f"""You are a supervisor agent evaluating answer clarity and completeness.

Query: "{query}"

Current Answer:
{answer[:1500]}

Evaluate:
1. Is the answer unclear or incomplete?
2. Are there gaps that web search could fill?
3. Is the supervisor (you) unsure about certain aspects?

Return ONLY valid JSON:
{{
  "is_unclear": true/false,
  "needs_search": true/false,
  "uncertainty_areas": ["..."],
  "search_queries": ["specific search queries to fill gaps"],
  "rationale": "brief explanation"
}}

Only suggest search if the answer is genuinely unclear or has significant gaps that search could address."""
    
    system_prompt = "You are a clarity evaluator. Assess if additional web search is needed. Return ONLY valid JSON."
    
    response = await call_agent(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL_LITE,
        temperature=0.2
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            evaluation = json.loads(response[json_start:json_end])
            logger.info(f"[GEMINI SUPERVISOR] Clarity check: unclear={evaluation.get('is_unclear', False)}, needs_search={evaluation.get('needs_search', False)}")
            return evaluation
        else:
            raise ValueError("Clarity check JSON not found")
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Clarity check parsing failed: {exc}")
        return {"is_unclear": False, "needs_search": False, "search_queries": []}


def gemini_supervisor_synthesize(query: str, medswin_answers: list, rag_contexts: list, search_contexts: list, breakdown: dict) -> str:
    """Wrapper to synthesize answer synchronously"""
    if not MCP_AVAILABLE:
        logger.warning("[GEMINI SUPERVISOR] MCP unavailable for synthesis, using simple concatenation")
        return "\n\n".join(medswin_answers)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                return nest_asyncio.run(gemini_supervisor_synthesize_async(query, medswin_answers, rag_contexts, search_contexts, breakdown))
            else:
                logger.error("[GEMINI SUPERVISOR] Nested synthesis failed: nest_asyncio not available")
        return loop.run_until_complete(gemini_supervisor_synthesize_async(query, medswin_answers, rag_contexts, search_contexts, breakdown))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Synthesis failed: {exc}")
        return "\n\n".join(medswin_answers)


def gemini_supervisor_challenge(query: str, current_answer: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> dict:
    """Wrapper to challenge answer synchronously"""
    if not MCP_AVAILABLE:
        return {"is_optimal": True, "completeness_score": 7, "accuracy_score": 7, "clarity_score": 7, "missing_aspects": [], "inaccuracies": [], "improvement_suggestions": [], "needs_more_context": False, "enhancement_instructions": ""}
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                return nest_asyncio.run(gemini_supervisor_challenge_async(query, current_answer, medswin_answers, rag_contexts, search_contexts))
            else:
                logger.error("[GEMINI SUPERVISOR] Nested challenge failed: nest_asyncio not available")
        return loop.run_until_complete(gemini_supervisor_challenge_async(query, current_answer, medswin_answers, rag_contexts, search_contexts))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Challenge failed: {exc}")
        return {"is_optimal": True, "completeness_score": 7, "accuracy_score": 7, "clarity_score": 7, "missing_aspects": [], "inaccuracies": [], "improvement_suggestions": [], "needs_more_context": False, "enhancement_instructions": ""}


def gemini_supervisor_enhance_answer(query: str, current_answer: str, enhancement_instructions: str, medswin_answers: list, rag_contexts: list, search_contexts: list) -> str:
    """Wrapper to enhance answer synchronously"""
    if not MCP_AVAILABLE:
        return current_answer
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                return nest_asyncio.run(gemini_supervisor_enhance_answer_async(query, current_answer, enhancement_instructions, medswin_answers, rag_contexts, search_contexts))
            else:
                logger.error("[GEMINI SUPERVISOR] Nested enhancement failed: nest_asyncio not available")
        return loop.run_until_complete(gemini_supervisor_enhance_answer_async(query, current_answer, enhancement_instructions, medswin_answers, rag_contexts, search_contexts))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Enhancement failed: {exc}")
        return current_answer


def gemini_supervisor_check_clarity(query: str, answer: str, use_web_search: bool) -> dict:
    """Wrapper to check clarity synchronously"""
    if not MCP_AVAILABLE or not use_web_search:
        return {"is_unclear": False, "needs_search": False, "search_queries": []}
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                return nest_asyncio.run(gemini_supervisor_check_clarity_async(query, answer, use_web_search))
            else:
                logger.error("[GEMINI SUPERVISOR] Nested clarity check failed: nest_asyncio not available")
        return loop.run_until_complete(gemini_supervisor_check_clarity_async(query, answer, use_web_search))
    except Exception as exc:
        logger.error(f"[GEMINI SUPERVISOR] Clarity check failed: {exc}")
        return {"is_unclear": False, "needs_search": False, "search_queries": []}


async def self_reflection_gemini(answer: str, query: str) -> dict:
    """Self-reflection using Gemini MCP"""
    reflection_prompt = f"""Evaluate this medical answer for quality and completeness:
Query: "{query}"
Answer: "{answer[:1000]}"
Evaluate:
1. Completeness: Does it address all aspects of the query?
2. Accuracy: Is the medical information accurate?
3. Clarity: Is it clear and well-structured?
4. Sources: Are sources cited appropriately?
5. Missing Information: What important information might be missing?
Respond in JSON:
{{
    "completeness_score": 0-10,
    "accuracy_score": 0-10,
    "clarity_score": 0-10,
    "overall_score": 0-10,
    "missing_aspects": ["..."],
    "improvement_suggestions": ["..."]
}}"""
    
    system_prompt = "You are a medical answer quality evaluator. Provide honest, constructive feedback."
    
    response = await call_agent(
        user_prompt=reflection_prompt,
        system_prompt=system_prompt,
        model=GEMINI_MODEL,
        temperature=0.3
    )
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            reflection = json.loads(response[json_start:json_end])
        else:
            reflection = {"overall_score": 7, "improvement_suggestions": []}
    except:
        reflection = {"overall_score": 7, "improvement_suggestions": []}
    
    logger.info(f"Self-reflection score: {reflection.get('overall_score', 'N/A')}")
    return reflection


def self_reflection(answer: str, query: str, reasoning: dict) -> dict:
    """Self-reflection: Evaluate answer quality and completeness"""
    if not MCP_AVAILABLE:
        logger.warning("Gemini MCP not available for reflection, using fallback")
        return {"overall_score": 7, "improvement_suggestions": []}
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                return nest_asyncio.run(self_reflection_gemini(answer, query))
            else:
                logger.error("Error in nested async reflection: nest_asyncio not available")
        else:
            return loop.run_until_complete(self_reflection_gemini(answer, query))
    except Exception as e:
        logger.error(f"Gemini MCP reflection error: {e}")
    
    return {"overall_score": 7, "improvement_suggestions": []}

