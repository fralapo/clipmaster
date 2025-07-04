import gradio as gr
from src.utils import get_recommended_whisper_model, load_api_keys
from src.logic import models_config
from src.events import register_event_handlers

def create_ui():
    saved_settings = load_api_keys()

    with gr.Blocks(title="ClipMaster", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ✂️ ClipMaster: AI-Powered Video Tools")
        
        status_box = gr.Textbox(label="Status", interactive=False, lines=3)
        
        # States for AI Clipping Workflow
        video_path_state = gr.State()
        processed_video_path_state = gr.State()
        transcription_path_state = gr.State()
        clips_state = gr.State([])

        with gr.Tabs():
            with gr.TabItem("AI Clipping Workflow"):
                with gr.Accordion("Step 1: Load Video", open=True) as step1_accordion:
                    input_type_radio = gr.Radio(["YouTube URL", "Local File"], label="Input Type", value="YouTube URL")
                    youtube_input = gr.Textbox(label="YouTube URL", visible=True)
                    local_file_input = gr.File(label="Local Video File", visible=False)
                    load_video_button = gr.Button("1. Load Video", variant="primary")
                
                with gr.Accordion("Step 2: Select Range & Transcribe", open=False) as step2_accordion:
                    with gr.Column(visible=True) as range_selector_col:
                        with gr.Row():
                            start_time_slider = gr.Slider(label="Start Time (seconds)", minimum=0, step=1, interactive=True)
                            end_time_slider = gr.Slider(label="End Time (seconds)", minimum=0, step=1, interactive=True)
                        with gr.Row():
                            language_dropdown = gr.Dropdown(["English", "Italian", "Spanish", "French", "German"], label="Video Language", value="English")
                            whisper_model = gr.Dropdown(["tiny", "base", "small", "medium", "large"], value=get_recommended_whisper_model(), label="Whisper Model")
                        process_range_button = gr.Button("2. Transcribe Selected Range", variant="primary")

                with gr.Accordion("Step 3: AI Analysis", open=False) as step3_accordion:
                    with gr.Row():
                        llm_service = gr.Dropdown(["gemini", "openai", "claude", "deepseek", "openrouter"], label="LLM Service", value="gemini")
                        api_key_input = gr.Textbox(label="API Key", type="password", placeholder="Enter key or load from Settings")
                    with gr.Row(visible=True) as standard_model_selector:
                        model_name_dropdown = gr.Radio(label="LLM Model", choices=models_config.get("gemini", {}).get("models", []), value=models_config.get("gemini", {}).get("default", ""))
                        manual_model_name_input = gr.Textbox(label="Or Enter Model Name Manually")
                    with gr.Column(visible=False) as openrouter_model_selector:
                        openrouter_model_input = gr.Textbox(label="OpenRouter Model Name", placeholder="e.g., google/gemini-flash-1.5")
                    context = gr.Textbox(lines=3, label="Optional Context")
                    with gr.Row():
                        num_clips = gr.Slider(1, 10, 5, step=1, label="Number of Clips")
                        duration = gr.Dropdown(["auto", "30s", "30-60s", "60-90s"], value="auto", label="Clip Duration")
                    analyze_button = gr.Button("3. Find Highlights & Create Previews", variant="primary", interactive=False)

                with gr.Accordion("Step 4: Review & Generate", open=False) as step4_accordion:
                    with gr.Row():
                        with gr.Column(scale=2):
                            main_preview_player = gr.Video(label="Full Video Preview", interactive=True)
                            clip_preview_player = gr.Video(label="Clip Preview", visible=False, interactive=True)
                        with gr.Column(scale=1):
                            gr.Markdown("### Edit Clip Details")
                            clip_selection_dropdown = gr.Dropdown(label="Select Clip to Edit", interactive=True)
                            with gr.Group():
                                edit_title = gr.Textbox(label="Title", interactive=True)
                                with gr.Row():
                                    edit_start_time = gr.Textbox(label="Start Time (ss or mm:ss)", interactive=True)
                                    edit_end_time = gr.Textbox(label="End Time (ss or mm:ss)", interactive=True)
                                edit_hashtags = gr.Textbox(label="Hashtags (comma-separated)", interactive=True)
                                edit_transcription = gr.Textbox(label="Transcription", interactive=False, lines=3)
                            with gr.Row():
                                preview_clip_button = gr.Button("Preview Clip")
                                save_clip_button = gr.Button("Save Changes", variant="primary")
                                delete_clip_button = gr.Button("Delete Clip", variant="stop")
                    
                    gr.Markdown("### Final Generation")
                    clips_dataframe = gr.Dataframe(
                        headers=["Score", "Title", "Start", "End", "Filename"], 
                        datatype=["number", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True
                    )
                    with gr.Row():
                        apply_reframe_checkbox = gr.Checkbox(label="Apply Auto Re-frame", value=True)
                    generate_button = gr.Button("4. Generate Final Clips", variant="primary", interactive=False)

            with gr.TabItem("Chapter Generator"):
                gr.Markdown("### YouTube Chapter Generator")
                with gr.Row():
                    with gr.Column():
                        chapter_input_type = gr.Radio(["YouTube URL", "Local File"], label="Input Type", value="YouTube URL")
                        chapter_youtube_input = gr.Textbox(label="YouTube URL", visible=True)
                        chapter_local_file_input = gr.File(label="Local Video File", visible=False)
                        with gr.Row():
                            chapter_language = gr.Dropdown(["English", "Italian", "Spanish", "French", "German"], label="Video Language", value="English")
                            chapter_whisper_model = gr.Dropdown(["tiny", "base", "small", "medium", "large"], value=get_recommended_whisper_model(), label="Whisper Model")
                        gr.Markdown("#### LLM Settings")
                        with gr.Row():
                            chapter_llm_service = gr.Dropdown(["gemini", "openai", "claude", "deepseek", "openrouter"], label="LLM Service", value="gemini")
                            chapter_api_key = gr.Textbox(label="API Key", type="password", placeholder="Enter key or load from Settings")
                        with gr.Row(visible=True) as chapter_standard_model_selector:
                            chapter_model_name = gr.Radio(label="LLM Model", choices=models_config.get("gemini", {}).get("models", []), value=models_config.get("gemini", {}).get("default", ""))
                            chapter_manual_model_name = gr.Textbox(label="Or Enter Model Name Manually")
                        with gr.Column(visible=False) as chapter_openrouter_model_selector:
                            chapter_openrouter_model = gr.Textbox(label="OpenRouter Model Name", placeholder="e.g., google/gemini-flash-1.5")
                        with gr.Row():
                            min_chapters_slider = gr.Slider(2, 10, 5, step=1, label="Min Chapters")
                            max_chapters_slider = gr.Slider(5, 25, 15, step=1, label="Max Chapters")
                        generate_chapters_button = gr.Button("Generate Chapters", variant="primary")
                    with gr.Column():
                        chapter_output = gr.Code(label="YouTube Chapters", language="markdown", interactive=True)

            with gr.TabItem("Video Captioner"):
                gr.Markdown("### Video Captioner")
                with gr.Row():
                    with gr.Column(scale=1):
                        caption_input_type = gr.Radio(["YouTube URL", "Local File"], label="Input Type", value="YouTube URL")
                        caption_youtube_input = gr.Textbox(label="YouTube URL", visible=True)
                        caption_local_file_input = gr.File(label="Local Video File", visible=False)
                        with gr.Row():
                            caption_language = gr.Dropdown(["English", "Italian", "Spanish", "French", "German"], label="Video Language", value="English")
                            caption_whisper_model = gr.Dropdown(["tiny", "base", "small", "medium", "large"], value=get_recommended_whisper_model(), label="Whisper Model")
                        gr.Markdown("#### Caption Style")
                        bg_color_picker = gr.ColorPicker(label="Background Color", value="#FFFFFF00")
                        hl_color_picker = gr.ColorPicker(label="Highlight Color", value="#FFE2A5DC")
                        txt_color_picker = gr.ColorPicker(label="Text Color", value="#000000")
                        generate_captions_button = gr.Button("Generate Captioned Video", variant="primary")
                    with gr.Column(scale=2):
                        captioned_video_output = gr.Video(label="Captioned Video Output")

            with gr.TabItem("Auto Re-frame Only"):
                gr.Markdown("### Batch Auto Re-framer")
                batch_video_input = gr.File(label="Upload Videos", file_count="multiple", type="filepath")
                batch_reframe_button = gr.Button("Start Re-framing", variant="primary")

            with gr.TabItem("Settings"):
                gr.Markdown("### API Key Management")
                gemini_key = gr.Textbox(label="Google Gemini API Key", value=saved_settings.get("GEMINI_API_KEY", ""), type="password")
                openai_key = gr.Textbox(label="OpenAI API Key", value=saved_settings.get("OPENAI_API_KEY", ""), type="password")
                claude_key = gr.Textbox(label="Anthropic Claude API Key", value=saved_settings.get("ANTHROPIC_API_KEY", ""), type="password")
                deepseek_key = gr.Textbox(label="DeepSeek API Key", value=saved_settings.get("DEEPSEEK_API_KEY", ""), type="password")
                openrouter_key = gr.Textbox(label="OpenRouter API Key", value=saved_settings.get("OPENROUTER_API_KEY", ""), type="password")
                gr.Markdown("### Whisper Settings")
                whisper_device_radio = gr.Radio(["Auto", "CPU", "GPU"], label="Whisper Processing Device", value=saved_settings.get("WHISPER_DEVICE", "Auto"))
                save_settings_button = gr.Button("Save Settings", variant="primary")
        
        gr.Markdown("""
        ---
        *Note: For optimal and free animated subtitles for your clips, consider using [withsubtitles.com](https://withsubtitles.com/).*
        """)

        components = {
            "status_box": status_box,
            "video_path_state": video_path_state,
            "processed_video_path_state": processed_video_path_state,
            "transcription_path_state": transcription_path_state,
            "clips_state": clips_state,
            "input_type_radio": input_type_radio,
            "youtube_input": youtube_input,
            "local_file_input": local_file_input,
            "load_video_button": load_video_button,
            "range_selector_col": range_selector_col,
            "start_time_slider": start_time_slider,
            "end_time_slider": end_time_slider,
            "language_dropdown": language_dropdown,
            "whisper_model": whisper_model,
            "process_range_button": process_range_button,
            "step1_accordion": step1_accordion,
            "step2_accordion": step2_accordion,
            "step3_accordion": step3_accordion,
            "step4_accordion": step4_accordion,
            "llm_service": llm_service,
            "api_key_input": api_key_input,
            "standard_model_selector": standard_model_selector,
            "model_name_dropdown": model_name_dropdown,
            "manual_model_name_input": manual_model_name_input,
            "openrouter_model_selector": openrouter_model_selector,
            "openrouter_model_input": openrouter_model_input,
            "context": context,
            "num_clips": num_clips,
            "duration": duration,
            "analyze_button": analyze_button,
            "main_preview_player": main_preview_player,
            "clip_preview_player": clip_preview_player,
            "clip_selection_dropdown": clip_selection_dropdown,
            "edit_title": edit_title,
            "edit_start_time": edit_start_time,
            "edit_end_time": edit_end_time,
            "edit_hashtags": edit_hashtags,
            "edit_transcription": edit_transcription,
            "preview_clip_button": preview_clip_button,
            "save_clip_button": save_clip_button,
            "delete_clip_button": delete_clip_button,
            "clips_dataframe": clips_dataframe,
            "apply_reframe_checkbox": apply_reframe_checkbox,
            "generate_button": generate_button,
            "chapter_input_type": chapter_input_type,
            "chapter_youtube_input": chapter_youtube_input,
            "chapter_local_file_input": chapter_local_file_input,
            "chapter_language": chapter_language,
            "chapter_whisper_model": chapter_whisper_model,
            "chapter_llm_service": chapter_llm_service,
            "chapter_api_key": chapter_api_key,
            "chapter_standard_model_selector": chapter_standard_model_selector,
            "chapter_model_name": chapter_model_name,
            "chapter_manual_model_name": chapter_manual_model_name,
            "chapter_openrouter_model_selector": chapter_openrouter_model_selector,
            "chapter_openrouter_model": chapter_openrouter_model,
            "min_chapters_slider": min_chapters_slider,
            "max_chapters_slider": max_chapters_slider,
            "generate_chapters_button": generate_chapters_button,
            "chapter_output": chapter_output,
            "caption_input_type": caption_input_type,
            "caption_youtube_input": caption_youtube_input,
            "caption_local_file_input": caption_local_file_input,
            "caption_language": caption_language,
            "caption_whisper_model": caption_whisper_model,
            "bg_color_picker": bg_color_picker,
            "hl_color_picker": hl_color_picker,
            "txt_color_picker": txt_color_picker,
            "generate_captions_button": generate_captions_button,
            "captioned_video_output": captioned_video_output,
            "batch_video_input": batch_video_input,
            "batch_reframe_button": batch_reframe_button,
            "gemini_key": gemini_key,
            "openai_key": openai_key,
            "claude_key": claude_key,
            "deepseek_key": deepseek_key,
            "openrouter_key": openrouter_key,
            "whisper_device_radio": whisper_device_radio,
            "save_settings_button": save_settings_button
        }
        
        register_event_handlers(components)

    return demo
