"""Gradio UI setup"""
import time
import gradio as gr
from config import TITLE, DESCRIPTION, CSS, MEDSWIN_MODELS, DEFAULT_MEDICAL_MODEL
from indexing import create_or_update_index
from pipeline import stream_chat
from voice import transcribe_audio, generate_speech


def create_demo():
    """Create and return Gradio demo interface"""
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(TITLE)
        gr.HTML(DESCRIPTION)
        
        with gr.Row(elem_classes="main-container"):
            with gr.Column(elem_classes="upload-section"):
                file_upload = gr.File(
                    file_count="multiple",
                    label="Drag and Drop Files Here",
                    file_types=[".pdf", ".txt", ".doc", ".docx", ".md", ".json", ".xml", ".csv"],
                    elem_id="file-upload"
                )
                upload_button = gr.Button("Upload & Index", elem_classes="upload-button")
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Upload files to start...",
                    interactive=False
                )
                file_info_output = gr.HTML(
                    label="File Information",
                    elem_classes="processing-info"
                )
                upload_button.click(
                    fn=create_or_update_index,
                    inputs=[file_upload],
                    outputs=[status_output, file_info_output]
                )
            
            with gr.Column(elem_classes="chatbot-container"):
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Chat with MedSwin... Type your question below.",
                    show_label=False,
                    type="messages"
                )
                with gr.Row(elem_classes="input-row"):
                    message_input = gr.Textbox(
                        placeholder="Type your medical question here...",
                        show_label=False,
                        container=False,
                        lines=1,
                        scale=10
                    )
                    mic_button = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="",
                        show_label=False,
                        container=False,
                        scale=1
                    )
                    submit_button = gr.Button("➤", elem_classes="submit-btn", scale=1)
                
                recording_timer = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
                    interactive=False,
                    visible=False,
                    container=False,
                    elem_classes="recording-timer"
                )
                
                recording_start_time = [None]
                
                def handle_recording_start():
                    """Called when recording starts"""
                    recording_start_time[0] = time.time()
                    return gr.update(visible=True, value="Recording... 0s")
                
                def handle_recording_stop(audio):
                    """Called when recording stops"""
                    recording_start_time[0] = None
                    if audio is None:
                        return gr.update(visible=False, value=""), ""
                    transcribed = transcribe_audio(audio)
                    return gr.update(visible=False, value=""), transcribed
                
                mic_button.start_recording(
                    fn=handle_recording_start,
                    outputs=[recording_timer]
                )
                
                mic_button.stop_recording(
                    fn=handle_recording_stop,
                    inputs=[mic_button],
                    outputs=[recording_timer, message_input]
                )
                
                with gr.Row(visible=False) as tts_row:
                    tts_text = gr.Textbox(visible=False)
                    tts_audio = gr.Audio(label="Generated Speech", visible=False)
                
                def generate_speech_from_chat(history):
                    """Extract last assistant message and generate speech"""
                    if not history or len(history) == 0:
                        return None
                    last_msg = history[-1]
                    if last_msg.get("role") == "assistant":
                        text = last_msg.get("content", "").replace(" 🔊", "").strip()
                        if text:
                            audio_path = generate_speech(text)
                            return audio_path
                    return None
                
                tts_button = gr.Button("🔊 Play Response", visible=False, size="sm")
                
                def update_tts_button(history):
                    if history and len(history) > 0 and history[-1].get("role") == "assistant":
                        return gr.update(visible=True)
                    return gr.update(visible=False)
                
                chatbot.change(
                    fn=update_tts_button,
                    inputs=[chatbot],
                    outputs=[tts_button]
                )
                
                tts_button.click(
                    fn=generate_speech_from_chat,
                    inputs=[chatbot],
                    outputs=[tts_audio]
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        disable_agentic_reasoning = gr.Checkbox(
                            value=False,
                            label="Disable agentic reasoning",
                            info="Use MedSwin model alone without agentic reasoning, RAG, or web search"
                        )
                        show_agentic_thought = gr.Button(
                            "Show agentic thought",
                            size="sm"
                        )
                    agentic_thoughts_box = gr.Textbox(
                        label="Agentic Thoughts",
                        placeholder="Internal thoughts from MedSwin and supervisor will appear here...",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        visible=False,
                        elem_classes="agentic-thoughts"
                    )
                    with gr.Row():
                        use_rag = gr.Checkbox(
                            value=False,
                            label="Enable Document RAG",
                            info="Answer based on uploaded documents (upload required)"
                        )
                        use_web_search = gr.Checkbox(
                            value=False,
                            label="Enable Web Search (MCP)",
                            info="Fetch knowledge from online medical resources"
                        )
                    
                    medical_model = gr.Radio(
                        choices=list(MEDSWIN_MODELS.keys()),
                        value=DEFAULT_MEDICAL_MODEL,
                        label="Medical Model",
                        info="MedSwin TA (default), others download on first use"
                    )
                    
                    system_prompt = gr.Textbox(
                        value="As a medical specialist, provide detailed and accurate answers based on the provided medical documents and context. Ensure all information is clinically accurate and cite sources when available.",
                        label="System Prompt",
                        lines=3
                    )
                    
                    with gr.Tab("Generation Parameters"):
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.2,
                            label="Temperature"
                        )
                        max_new_tokens = gr.Slider(
                            minimum=512,
                            maximum=4096,
                            step=128,
                            value=2048,
                            label="Max New Tokens",
                            info="Increased for medical models to prevent early stopping"
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.7,
                            label="Top P"
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                            label="Top K"
                        )
                        penalty = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            value=1.2,
                            label="Repetition Penalty"
                        )
                    
                    with gr.Tab("Retrieval Parameters"):
                        retriever_k = gr.Slider(
                            minimum=5,
                            maximum=30,
                            step=1,
                            value=15,
                            label="Initial Retrieval Size (Top K)"
                        )
                        merge_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            step=0.1,
                            value=0.5,
                            label="Merge Threshold (lower = more merging)"
                        )
                
                show_thoughts_state = gr.State(value=False)
                
                def toggle_thoughts_box(current_state):
                    """Toggle visibility of agentic thoughts box"""
                    new_state = not current_state
                    return gr.update(visible=new_state), new_state
                
                show_agentic_thought.click(
                    fn=toggle_thoughts_box,
                    inputs=[show_thoughts_state],
                    outputs=[agentic_thoughts_box, show_thoughts_state]
                )
                
                submit_button.click(
                    fn=stream_chat,
                    inputs=[
                        message_input,
                        chatbot,
                        system_prompt,
                        temperature,
                        max_new_tokens,
                        top_p,
                        top_k,
                        penalty,
                        retriever_k,
                        merge_threshold,
                        use_rag,
                        medical_model,
                        use_web_search,
                        disable_agentic_reasoning,
                        show_thoughts_state
                    ],
                    outputs=[chatbot, agentic_thoughts_box]
                )
                
                message_input.submit(
                    fn=stream_chat,
                    inputs=[
                        message_input,
                        chatbot,
                        system_prompt,
                        temperature,
                        max_new_tokens,
                        top_p,
                        top_k,
                        penalty,
                        retriever_k,
                        merge_threshold,
                        use_rag,
                        medical_model,
                        use_web_search,
                        disable_agentic_reasoning,
                        show_thoughts_state
                    ],
                    outputs=[chatbot, agentic_thoughts_box]
                )
    
    return demo

