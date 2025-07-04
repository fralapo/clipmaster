import gradio as gr
from src import logic

def register_event_handlers(components):
    
    def toggle_input_type(choice):
        return gr.update(visible=choice == "YouTube URL"), gr.update(visible=choice == "Local File")

    # AI Clipping Workflow
    components["input_type_radio"].change(toggle_input_type, components["input_type_radio"], [components["youtube_input"], components["local_file_input"]])
    components["load_video_button"].click(
        fn=logic.load_video_for_clipping, 
        inputs=[components["input_type_radio"], components["youtube_input"], components["local_file_input"]], 
        outputs=[components["video_path_state"], components["start_time_slider"], components["end_time_slider"], components["step1_accordion"], components["step2_accordion"]]
    )
    components["process_range_button"].click(
        fn=logic.process_range_and_transcribe, 
        inputs=[components["video_path_state"], components["start_time_slider"], components["end_time_slider"], components["whisper_model"], components["language_dropdown"]], 
        outputs=[components["processed_video_path_state"], components["transcription_path_state"], components["status_box"], components["step3_accordion"], components["analyze_button"]]
    )
    components["analyze_button"].click(
        fn=logic.run_analysis_and_create_previews, 
        inputs=[
            components["processed_video_path_state"], components["transcription_path_state"], components["llm_service"], 
            components["api_key_input"], components["model_name_dropdown"], components["manual_model_name_input"], 
            components["context"], components["num_clips"], components["duration"], components["language_dropdown"]
        ], 
        outputs=[
            components["clips_state"], components["status_box"], components["clips_dataframe"], 
            components["step4_accordion"], components["generate_button"], components["main_preview_player"], 
            components["clip_selection_dropdown"]
        ]
    )
    components["clip_selection_dropdown"].change(
        fn=logic.select_clip,
        inputs=[components["clips_state"], components["clip_selection_dropdown"]],
        outputs=[
            components["edit_title"], components["edit_start_time"], components["edit_end_time"],
            components["edit_hashtags"], components["edit_transcription"]
        ]
    )
    components["save_clip_button"].click(
        fn=logic.update_clip,
        inputs=[
            components["clips_state"], components["clip_selection_dropdown"], components["edit_title"],
            components["edit_start_time"], components["edit_end_time"], components["edit_hashtags"]
        ],
        outputs=[components["clips_state"], components["clips_dataframe"], components["status_box"]]
    )
    components["delete_clip_button"].click(
        fn=logic.delete_clip,
        inputs=[components["clips_state"], components["clip_selection_dropdown"]],
        outputs=[components["clips_state"], components["clips_dataframe"], components["clip_selection_dropdown"], components["status_box"]]
    )
    components["preview_clip_button"].click(
        fn=logic.preview_clip,
        inputs=[components["video_path_state"], components["clips_state"], components["clip_selection_dropdown"]],
        outputs=[components["clip_preview_player"]]
    )
    components["generate_button"].click(
        fn=logic.run_clip_generation, 
        inputs=[components["processed_video_path_state"], components["clips_dataframe"], components["apply_reframe_checkbox"]], 
        outputs=[components["status_box"]]
    )

    # Chapter Generator
    components["chapter_input_type"].change(toggle_input_type, components["chapter_input_type"], [components["chapter_youtube_input"], components["chapter_local_file_input"]])
    components["generate_chapters_button"].click(
        fn=logic.run_chapter_generation_logic, 
        inputs=[
            components["chapter_input_type"], components["chapter_youtube_input"], components["chapter_local_file_input"], 
            components["chapter_whisper_model"], components["chapter_language"], components["chapter_llm_service"], 
            components["chapter_api_key"], components["chapter_model_name"], components["chapter_manual_model_name"], 
            components["min_chapters_slider"], components["max_chapters_slider"]
        ], 
        outputs=[components["chapter_output"]]
    )

    # Video Captioner
    components["caption_input_type"].change(toggle_input_type, components["caption_input_type"], [components["caption_youtube_input"], components["caption_local_file_input"]])
    components["generate_captions_button"].click(
        fn=logic.run_captioning_logic, 
        inputs=[
            components["caption_input_type"], components["caption_youtube_input"], components["caption_local_file_input"], 
            components["caption_whisper_model"], components["caption_language"], components["bg_color_picker"], 
            components["hl_color_picker"], components["txt_color_picker"]
        ], 
        outputs=[components["captioned_video_output"]]
    )

    # Auto Re-frame Only
    components["batch_reframe_button"].click(fn=logic.run_batch_reframe, inputs=[components["batch_video_input"]], outputs=[components["status_box"]])

    # Settings
    components["save_settings_button"].click(
        fn=logic.handle_save_api_keys, 
        inputs=[
            components["gemini_key"], components["openai_key"], components["claude_key"], 
            components["deepseek_key"], components["openrouter_key"], components["whisper_device_radio"]
        ]
    )
    components["llm_service"].change(
        fn=logic.update_model_ui, 
        inputs=components["llm_service"], 
        outputs=[
            components["model_name_dropdown"], components["standard_model_selector"], 
            components["openrouter_model_selector"], components["openrouter_model_input"], components["api_key_input"]
        ]
    )
    components["chapter_llm_service"].change(
        fn=logic.update_model_ui, 
        inputs=components["chapter_llm_service"], 
        outputs=[
            components["chapter_model_name"], components["chapter_standard_model_selector"], 
            components["chapter_openrouter_model_selector"], components["chapter_openrouter_model"], components["chapter_api_key"]
        ]
    )
