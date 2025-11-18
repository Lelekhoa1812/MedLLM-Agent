"""Main chat pipeline - stream_chat function"""
import os
import json
import time
import logging
import concurrent.futures
import gradio as gr
import spaces
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.retrievers import AutoMergingRetriever
from logger import logger, ThoughtCaptureHandler
from models import initialize_medical_model, get_or_create_embed_model
from utils import detect_language, translate_text, format_url_as_domain
from search import search_web, summarize_web_content
from reasoning import autonomous_reasoning, create_execution_plan, autonomous_execution_strategy
from supervisor import (
    gemini_supervisor_breakdown, gemini_supervisor_search_strategies,
    gemini_supervisor_rag_brainstorm, execute_medswin_task,
    gemini_supervisor_synthesize, gemini_supervisor_challenge,
    gemini_supervisor_enhance_answer, gemini_supervisor_check_clarity
)


@spaces.GPU(max_duration=120)
def stream_chat(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    penalty: float,
    retriever_k: int,
    merge_threshold: float,
    use_rag: bool,
    medical_model: str,
    use_web_search: bool,
    disable_agentic_reasoning: bool,
    show_thoughts: bool,
    request: gr.Request
):
    """Main chat pipeline implementing MAC architecture"""
    if not request:
        yield history + [{"role": "assistant", "content": "Session initialization failed. Please refresh the page."}], ""
        return
    
    thought_handler = None
    if show_thoughts:
        thought_handler = ThoughtCaptureHandler()
        thought_handler.setLevel(logging.INFO)
        thought_handler.clear()
        logger.addHandler(thought_handler)
    
    session_start = time.time()
    soft_timeout = 100
    hard_timeout = 118
    
    def elapsed():
        return time.time() - session_start
    
    user_id = request.session_hash
    index_dir = f"./{user_id}_index"
    has_rag_index = os.path.exists(index_dir)
    
    original_lang = detect_language(message)
    original_message = message
    needs_translation = original_lang != "en"
    
    pipeline_diagnostics = {
        "reasoning": None,
        "plan": None,
        "strategy_decisions": [],
        "stage_metrics": {},
        "search": {"strategies": [], "total_results": 0}
    }
    
    def record_stage(stage_name: str, start_time: float):
        pipeline_diagnostics["stage_metrics"][stage_name] = round(time.time() - start_time, 3)
    
    translation_stage_start = time.time()
    if needs_translation:
        logger.info(f"[GEMINI SUPERVISOR] Detected non-English language: {original_lang}, translating...")
        message = translate_text(message, target_lang="en", source_lang=original_lang)
        logger.info(f"[GEMINI SUPERVISOR] Translated query: {message[:100]}...")
    record_stage("translation", translation_stage_start)
    
    final_use_rag = use_rag and has_rag_index and not disable_agentic_reasoning
    final_use_web_search = use_web_search and not disable_agentic_reasoning
    
    plan = None
    if not disable_agentic_reasoning:
        reasoning_stage_start = time.time()
        reasoning = autonomous_reasoning(message, history)
        record_stage("autonomous_reasoning", reasoning_stage_start)
        pipeline_diagnostics["reasoning"] = reasoning
        plan = create_execution_plan(reasoning, message, has_rag_index)
        pipeline_diagnostics["plan"] = plan
        execution_strategy = autonomous_execution_strategy(
            reasoning, plan, final_use_rag, final_use_web_search, has_rag_index
        )
        
        if final_use_rag and not reasoning.get("requires_rag", True):
            final_use_rag = False
            pipeline_diagnostics["strategy_decisions"].append("Skipped RAG per autonomous reasoning")
        elif not final_use_rag and reasoning.get("requires_rag", True) and not has_rag_index:
            pipeline_diagnostics["strategy_decisions"].append("Reasoning wanted RAG but no index available")
        
        if final_use_web_search and not reasoning.get("requires_web_search", False):
            final_use_web_search = False
            pipeline_diagnostics["strategy_decisions"].append("Skipped web search per autonomous reasoning")
        elif not final_use_web_search and reasoning.get("requires_web_search", False):
            if not use_web_search:
                pipeline_diagnostics["strategy_decisions"].append("User disabled web search despite reasoning request")
            else:
                pipeline_diagnostics["strategy_decisions"].append("Web search requested by reasoning but disabled by mode")
    else:
        pipeline_diagnostics["strategy_decisions"].append("Agentic reasoning disabled by user")
    
    if disable_agentic_reasoning:
        logger.info("[MAC] Agentic reasoning disabled - using MedSwin alone")
        breakdown = {
            "sub_topics": [
                {"id": 1, "topic": "Answer", "instruction": message, "expected_tokens": 400, "priority": "high", "approach": "direct answer"}
            ],
            "strategy": "Direct answer",
            "exploration_note": "Direct mode - no breakdown"
        }
    else:
        logger.info("[GEMINI SUPERVISOR] Breaking query into sub-topics...")
        breakdown = gemini_supervisor_breakdown(message, final_use_rag, final_use_web_search, elapsed(), max_duration=120)
        logger.info(f"[GEMINI SUPERVISOR] Created {len(breakdown.get('sub_topics', []))} sub-topics")
    
    search_contexts = []
    web_urls = []
    if final_use_web_search:
        search_stage_start = time.time()
        logger.info("[GEMINI SUPERVISOR] Search mode: Creating search strategies...")
        search_strategies = gemini_supervisor_search_strategies(message, elapsed())
        
        all_search_results = []
        strategy_jobs = []
        for strategy in search_strategies.get("search_strategies", [])[:4]:
            search_query = strategy.get("strategy", message)
            target_sources = strategy.get("target_sources", 2)
            strategy_jobs.append({
                "query": search_query,
                "target_sources": target_sources,
                "meta": strategy
            })
        
        def execute_search(job):
            job_start = time.time()
            try:
                results = search_web(job["query"], max_results=job["target_sources"])
                duration = time.time() - job_start
                return results, duration, None
            except Exception as exc:
                return [], time.time() - job_start, exc
        
        def record_search_diag(job, duration, results_count, error=None):
            entry = {
                "query": job["query"],
                "target_sources": job["target_sources"],
                "duration": round(duration, 3),
                "results": results_count
            }
            if error:
                entry["error"] = str(error)
            pipeline_diagnostics["search"]["strategies"].append(entry)
        
        if strategy_jobs:
            max_workers = min(len(strategy_jobs), 4)
            if len(strategy_jobs) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {executor.submit(execute_search, job): job for job in strategy_jobs}
                    for future in concurrent.futures.as_completed(future_map):
                        job = future_map[future]
                        try:
                            results, duration, error = future.result()
                        except Exception as exc:
                            results, duration, error = [], 0.0, exc
                        record_search_diag(job, duration, len(results), error)
                        if not error and results:
                            all_search_results.extend(results)
                            web_urls.extend([r.get('url', '') for r in results if r.get('url')])
            else:
                job = strategy_jobs[0]
                results, duration, error = execute_search(job)
                record_search_diag(job, duration, len(results), error)
                if not error and results:
                    all_search_results.extend(results)
                    web_urls.extend([r.get('url', '') for r in results if r.get('url')])
        else:
            pipeline_diagnostics["strategy_decisions"].append("No viable web search strategies returned")
        
        pipeline_diagnostics["search"]["total_results"] = len(all_search_results)
        
        if all_search_results:
            logger.info(f"[GEMINI SUPERVISOR] Summarizing {len(all_search_results)} search results...")
            search_summary = summarize_web_content(all_search_results, message)
            if search_summary:
                search_contexts.append(search_summary)
                logger.info(f"[GEMINI SUPERVISOR] Search summary created: {len(search_summary)} chars")
        record_stage("web_search", search_stage_start)
    
    rag_contexts = []
    if final_use_rag and has_rag_index:
        rag_stage_start = time.time()
        if elapsed() >= soft_timeout - 10:
            logger.warning("[GEMINI SUPERVISOR] Skipping RAG due to time pressure")
            final_use_rag = False
        else:
            logger.info("[GEMINI SUPERVISOR] RAG mode: Retrieving documents...")
            embed_model = get_or_create_embed_model()
            Settings.embed_model = embed_model
            storage_context = StorageContext.from_defaults(persist_dir=index_dir)
            index = load_index_from_storage(storage_context, settings=Settings)
            base_retriever = index.as_retriever(similarity_top_k=retriever_k)
            auto_merging_retriever = AutoMergingRetriever(
                base_retriever,
                storage_context=storage_context,
                simple_ratio_thresh=merge_threshold,
                verbose=False
            )
            merged_nodes = auto_merging_retriever.retrieve(message)
            retrieved_docs = "\n\n".join([n.node.text for n in merged_nodes])
            logger.info(f"[GEMINI SUPERVISOR] Retrieved {len(merged_nodes)} document nodes")
            
            logger.info("[GEMINI SUPERVISOR] Brainstorming RAG contexts...")
            rag_brainstorm = gemini_supervisor_rag_brainstorm(message, retrieved_docs, elapsed())
            rag_contexts = [ctx.get("context", "") for ctx in rag_brainstorm.get("contexts", [])]
            logger.info(f"[GEMINI SUPERVISOR] Created {len(rag_contexts)} RAG contexts")
        record_stage("rag_retrieval", rag_stage_start)
    
    medical_model_obj, medical_tokenizer = initialize_medical_model(medical_model)
    
    base_system_prompt = system_prompt if system_prompt else "As a medical specialist, provide clinical and concise answers. Use Markdown format with bullet points. Do not use tables."
    
    combined_context = ""
    if rag_contexts:
        combined_context += "Document Context:\n" + "\n\n".join(rag_contexts[:4])
    if search_contexts:
        if combined_context:
            combined_context += "\n\n"
        combined_context += "Web Search Context:\n" + "\n\n".join(search_contexts)
    
    logger.info(f"[MEDSWIN] Executing {len(breakdown.get('sub_topics', []))} tasks sequentially...")
    medswin_answers = []
    
    updated_history = history + [
        {"role": "user", "content": original_message},
        {"role": "assistant", "content": ""}
    ]
    thoughts_text = thought_handler.get_thoughts() if thought_handler else ""
    yield updated_history, thoughts_text
    
    medswin_stage_start = time.time()
    for idx, sub_topic in enumerate(breakdown.get("sub_topics", []), 1):
        if elapsed() >= hard_timeout - 5:
            logger.warning(f"[MEDSWIN] Time limit approaching, stopping at task {idx}")
            break
        
        task_instruction = sub_topic.get("instruction", "")
        topic_name = sub_topic.get("topic", f"Topic {idx}")
        priority = sub_topic.get("priority", "medium")
        
        logger.info(f"[MEDSWIN] Executing task {idx}/{len(breakdown.get('sub_topics', []))}: {topic_name} (priority: {priority})")
        
        task_context = combined_context
        if len(rag_contexts) > 1 and idx <= len(rag_contexts):
            task_context = rag_contexts[idx - 1] if idx <= len(rag_contexts) else combined_context
        
        try:
            task_answer = execute_medswin_task(
                medical_model_obj=medical_model_obj,
                medical_tokenizer=medical_tokenizer,
                task_instruction=task_instruction,
                context=task_context if task_context else "",
                system_prompt_base=base_system_prompt,
                temperature=temperature,
                max_new_tokens=min(max_new_tokens, 800),
                top_p=top_p,
                top_k=top_k,
                penalty=penalty
            )
            
            formatted_answer = f"## {topic_name}\n\n{task_answer}"
            medswin_answers.append(formatted_answer)
            logger.info(f"[MEDSWIN] Task {idx} completed: {len(task_answer)} chars")
            
            partial_final = "\n\n".join(medswin_answers)
            updated_history[-1]["content"] = partial_final
            thoughts_text = thought_handler.get_thoughts() if thought_handler else ""
            yield updated_history, thoughts_text
    
        except Exception as e:
            logger.error(f"[MEDSWIN] Task {idx} failed: {e}")
            continue
    record_stage("medswin_tasks", medswin_stage_start)
    
    logger.info("[GEMINI SUPERVISOR] Synthesizing final answer from all MedSwin responses...")
    raw_medswin_answers = [ans.split('\n\n', 1)[1] if '\n\n' in ans else ans for ans in medswin_answers]
    synthesis_stage_start = time.time()
    final_answer = gemini_supervisor_synthesize(message, raw_medswin_answers, rag_contexts, search_contexts, breakdown)
    record_stage("synthesis", synthesis_stage_start)
    
    if not final_answer or len(final_answer.strip()) < 50:
        logger.warning("[GEMINI SUPERVISOR] Synthesis failed or too short, using concatenation")
        final_answer = "\n\n".join(medswin_answers) if medswin_answers else "I apologize, but I was unable to generate a response."
    
    if "|" in final_answer and "---" in final_answer:
        logger.warning("[MEDSWIN] Final answer contains tables, converting to bullets")
        lines = final_answer.split('\n')
        cleaned_lines = []
        for line in lines:
            if '|' in line and '---' not in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    cleaned_lines.append(f"- {' / '.join(cells)}")
            elif '---' not in line:
                cleaned_lines.append(line)
        final_answer = '\n'.join(cleaned_lines)
    
    max_challenge_iterations = 2
    challenge_iteration = 0
    challenge_stage_start = time.time()
    
    while challenge_iteration < max_challenge_iterations and elapsed() < soft_timeout - 15:
        challenge_iteration += 1
        logger.info(f"[GEMINI SUPERVISOR] Challenge iteration {challenge_iteration}/{max_challenge_iterations}...")
        
        evaluation = gemini_supervisor_challenge(message, final_answer, raw_medswin_answers, rag_contexts, search_contexts)
        
        if evaluation.get("is_optimal", False):
            logger.info(f"[GEMINI SUPERVISOR] Answer confirmed optimal after {challenge_iteration} iteration(s)")
            break
        
        enhancement_instructions = evaluation.get("enhancement_instructions", "")
        if not enhancement_instructions:
            logger.info("[GEMINI SUPERVISOR] No enhancement instructions, considering answer optimal")
            break
        
        logger.info(f"[GEMINI SUPERVISOR] Enhancing answer based on feedback...")
        enhanced_answer = gemini_supervisor_enhance_answer(
            message, final_answer, enhancement_instructions, raw_medswin_answers, rag_contexts, search_contexts
        )
        
        if enhanced_answer and len(enhanced_answer.strip()) > len(final_answer.strip()) * 0.8:
            final_answer = enhanced_answer
            logger.info(f"[GEMINI SUPERVISOR] Answer enhanced (new length: {len(final_answer)} chars)")
        else:
            logger.info("[GEMINI SUPERVISOR] Enhancement did not improve answer significantly, stopping")
            break
    record_stage("challenge_loop", challenge_stage_start)
    
    if final_use_web_search and elapsed() < soft_timeout - 10:
        logger.info("[GEMINI SUPERVISOR] Checking if additional search is needed...")
        clarity_stage_start = time.time()
        clarity_check = gemini_supervisor_check_clarity(message, final_answer, final_use_web_search)
        record_stage("clarity_check", clarity_stage_start)
        
        if clarity_check.get("needs_search", False) and clarity_check.get("search_queries"):
            logger.info(f"[GEMINI SUPERVISOR] Triggering additional search: {clarity_check.get('search_queries', [])}")
            additional_search_results = []
            followup_stage_start = time.time()
            for search_query in clarity_check.get("search_queries", [])[:3]:
                if elapsed() >= soft_timeout - 5:
                    break
                extra_start = time.time()
                results = search_web(search_query, max_results=2)
                extra_duration = time.time() - extra_start
                pipeline_diagnostics["search"]["strategies"].append({
                    "query": search_query,
                    "target_sources": 2,
                    "duration": round(extra_duration, 3),
                    "results": len(results),
                    "type": "followup"
                })
                additional_search_results.extend(results)
                web_urls.extend([r.get('url', '') for r in results if r.get('url')])
            
            if additional_search_results:
                pipeline_diagnostics["search"]["total_results"] += len(additional_search_results)
                logger.info(f"[GEMINI SUPERVISOR] Summarizing {len(additional_search_results)} additional search results...")
                additional_summary = summarize_web_content(additional_search_results, message)
                if additional_summary:
                    search_contexts.append(additional_summary)
                    logger.info("[GEMINI SUPERVISOR] Enhancing answer with additional search context...")
                    enhanced_with_search = gemini_supervisor_enhance_answer(
                        message, final_answer,
                        f"Incorporate the following additional information from web search: {additional_summary}",
                        raw_medswin_answers, rag_contexts, search_contexts
                    )
                    if enhanced_with_search and len(enhanced_with_search.strip()) > 50:
                        final_answer = enhanced_with_search
                        logger.info("[GEMINI SUPERVISOR] Answer enhanced with additional search context")
            record_stage("followup_search", followup_stage_start)
    
    citations_text = ""
    
    if needs_translation and final_answer:
        logger.info(f"[GEMINI SUPERVISOR] Translating response back to {original_lang}...")
        final_answer = translate_text(final_answer, target_lang=original_lang, source_lang="en")
    
    if web_urls:
        unique_urls = list(dict.fromkeys(web_urls))
        citation_links = []
        for url in unique_urls[:5]:
            domain = format_url_as_domain(url)
            if domain:
                citation_links.append(f"[{domain}]({url})")
        
        if citation_links:
            citations_text = "\n\n**Sources:** " + ", ".join(citation_links)
    
    speaker_icon = ' 🔊'
    final_answer_with_metadata = final_answer + citations_text + speaker_icon
    
    updated_history[-1]["content"] = final_answer_with_metadata
    thoughts_text = thought_handler.get_thoughts() if thought_handler else ""
    yield updated_history, thoughts_text
    
    if thought_handler:
        logger.removeHandler(thought_handler)
    
    diag_summary = {
        "stage_metrics": pipeline_diagnostics["stage_metrics"],
        "decisions": pipeline_diagnostics["strategy_decisions"],
        "search": pipeline_diagnostics["search"],
    }
    try:
        logger.info(f"[MAC] Diagnostics summary: {json.dumps(diag_summary)[:1200]}")
    except Exception:
        logger.info(f"[MAC] Diagnostics summary (non-serializable)")
    logger.info(f"[MAC] Final answer generated: {len(final_answer)} chars, {len(breakdown.get('sub_topics', []))} tasks completed")

