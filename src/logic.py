import os
import json
import logging
import pandas as pd
import gradio as gr
from src.video_handler import download_video, extract_audio
from src.transcriber import transcribe_audio
from src.analyzer import analyze_transcription
from src.reframe import reframe_video
from src.processor import cut_clip_simple, create_preview_clip, cut_video_for_processing
from src.utils import get_recommended_whisper_model, cleanup_temp_files, save_api_keys, load_api_keys, sanitize_filename
from src.chapter_generator import LLMChapterGenerator, get_video_duration, format_chapters_for_youtube
from src.captioner import process_segments_for_captions, caption_video

# --- Config Loading ---
def load_models_config():
    config_path = os.path.join(os.path.dirname(__file__), 'models_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)
models_config = load_models_config()

# --- Logic Functions ---
def format_time(seconds):
    if seconds is None: return "00:00"
    seconds = int(seconds)
    minutes = seconds // 60
    seconds %= 60
    return f"{minutes:02d}:{seconds:02d}"

def deformat_time(time_str):
    if isinstance(time_str, (int, float)):
        return time_str
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
    except:
        return 0
    return 0

def get_transcription_for_clip(full_transcription, start_time, end_time):
    text = ""
    for segment in full_transcription.get('segments', []):
        for word in segment.get('words', []):
            if word.get('start') is not None and word.get('end') is not None:
                if start_time <= word['start'] <= end_time:
                    text += word['word'] + " "
    return text.strip()

def load_video_for_clipping(input_type, youtube_url, local_file, progress=gr.Progress()):
    progress(0, desc="Loading video...")
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if input_type == "YouTube URL" and youtube_url:
        video_path = download_video(youtube_url, temp_dir=temp_dir, progress=progress)
    elif input_type == "Local File" and local_file is not None:
        video_path = local_file.name
    else:
        raise gr.Error("Please provide a valid input for the selected type.")
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Failed to load video.")
    
    progress(0.8, desc="Getting video duration...")
    duration_sec = get_video_duration(video_path)
    if duration_sec == 0:
        raise gr.Error("Could not determine video duration.")
    
    return (
        video_path, 
        gr.update(maximum=duration_sec, value=0), 
        gr.update(maximum=duration_sec, value=duration_sec),
        gr.update(open=False), # Close Step 1
        gr.update(open=True)  # Open Step 2
    )

def process_range_and_transcribe(video_path, start_sec, end_sec, whisper_model, language, progress=gr.Progress()):
    if start_sec >= end_sec:
        raise gr.Error("Start time must be less than end time.")

    total_duration = get_video_duration(video_path)
    # Check if the user has selected the entire video duration (with a small tolerance)
    is_full_range = start_sec == 0 and end_sec >= total_duration - 1

    temp_dir = "temp"
    
    if is_full_range:
        progress(0, desc="Full range selected. Extracting full audio...")
        audio_path = extract_audio(video_path, temp_dir=temp_dir)
        if not audio_path:
            raise gr.Error("Failed to extract full audio.")
    else:
        progress(0, desc="Partial range selected. Extracting audio segment...")
        from src.video_handler import extract_audio_segment
        temp_audio_path = os.path.join(temp_dir, f"temp_audio_segment_{os.path.basename(video_path)}.mp3")
        audio_path = extract_audio_segment(video_path, start_sec, end_sec, temp_audio_path)
        if not audio_path:
            raise gr.Error("Failed to extract audio from the selected range.")

    progress(0.5, desc="Starting transcription...")
    saved_settings = load_api_keys()
    device_setting = saved_settings.get("WHISPER_DEVICE", "auto").lower()
    device = "cuda" if device_setting == "gpu" else device_setting

    transcription_path = transcribe_audio(audio_path, model_name=whisper_model, language=language, device=device)
    
    if not transcription_path:
        raise gr.Error("Transcription failed.")
    
    progress(1, desc="Transcription complete.")
    
    # The next step always uses the original, full video path for clipping later.
    return video_path, transcription_path, "Transcription of selected range is complete. Ready for Analysis.", gr.update(open=True), gr.update(interactive=True)

def run_analysis_and_create_previews(video_path, transcription_path, llm_service, api_key, model_name, manual_model_name, context, num_clips, duration, language, progress=gr.Progress()):
    progress(0, desc="Starting analysis...")
    final_model_name = manual_model_name if manual_model_name and llm_service != "openrouter" else model_name
    if llm_service == "openrouter":
        final_model_name = manual_model_name
    
    suggestions_path = analyze_transcription(transcription_path, context, num_clips, llm_service, api_key, duration, final_model_name, language)
    if not suggestions_path:
        raise gr.Error("Content analysis failed.")
        
    with open(suggestions_path, 'r', encoding='utf-8') as f:
        suggestions = json.load(f)
    with open(transcription_path, 'r', encoding='utf-8') as f:
        full_transcription = json.load(f)
    
    progress(0.5, desc="Analysis complete. Generating previews...")
    
    clips = suggestions.get('clips', [])
    for i, clip in enumerate(clips):
        clip['id'] = i
        clip['filename'] = sanitize_filename(f"{clip['title']} {' '.join(clip['hashtags'])}")
        clip['transcription'] = get_transcription_for_clip(full_transcription, clip['start_time'], clip['end_time'])

    clip_titles = [f"Clip {i+1}: {clip['title']}" for i, clip in enumerate(clips)]
    df = pd.DataFrame([[c.get('score', 'N/A'), c['title'], format_time(c['start_time']), format_time(c['end_time']), c['filename']] for c in clips], columns=["Score", "Title", "Start", "End", "Filename"])
    
    progress(1, desc="Previews generated.")
    return clips, "Previews ready.", df, gr.update(open=True), gr.update(interactive=True), video_path, gr.update(choices=clip_titles, value=clip_titles[0] if clip_titles else None)

def select_clip(clips_state, selection):
    if not selection or not clips_state:
        return "", "", "", "", ""
    
    clip_index = int(selection.split(":")[0].replace("Clip ", "")) - 1
    clip = clips_state[clip_index]
    
    return (
        clip['title'],
        format_time(clip['start_time']),
        format_time(clip['end_time']),
        ", ".join(clip['hashtags']),
        clip['transcription']
    )

def update_clip(clips_state, selection, title, start_time, end_time, hashtags):
    if not selection or not clips_state:
        return clips_state, "No clip selected."

    clip_index = int(selection.split(":")[0].replace("Clip ", "")) - 1
    
    clips_state[clip_index]['title'] = title
    clips_state[clip_index]['start_time'] = deformat_time(start_time)
    clips_state[clip_index]['end_time'] = deformat_time(end_time)
    clips_state[clip_index]['hashtags'] = [h.strip() for h in hashtags.split(',')]
    clips_state[clip_index]['filename'] = sanitize_filename(f"{title} {' '.join(clips_state[clip_index]['hashtags'])}")
    
    df = pd.DataFrame([[c.get('score', 'N/A'), c['title'], format_time(c['start_time']), format_time(c['end_time']), c['filename']] for c in clips_state], columns=["Score", "Title", "Start", "End", "Filename"])
    
    return clips_state, df, "Clip updated successfully."

def delete_clip(clips_state, selection):
    if not selection or not clips_state:
        return clips_state, "No clip selected."

    clip_index = int(selection.split(":")[0].replace("Clip ", "")) - 1
    del clips_state[clip_index]
    
    clip_titles = [f"Clip {i+1}: {clip['title']}" for i, clip in enumerate(clips_state)]
    df = pd.DataFrame([[c.get('score', 'N/A'), c['title'], format_time(c['start_time']), format_time(c['end_time']), c['filename']] for c in clips_state], columns=["Score", "Title", "Start", "End", "Filename"])
    
    return clips_state, df, gr.update(choices=clip_titles, value=clip_titles[0] if clip_titles else None), "Clip deleted."

def preview_clip(video_path, clips_state, selection):
    if not selection or not clips_state:
        return gr.update(visible=False)

    clip_index = int(selection.split(":")[0].replace("Clip ", "")) - 1
    clip = clips_state[clip_index]
    
    temp_dir = "temp"
    preview_output_path = os.path.join(temp_dir, f"preview_{clip['id']}.mp4")
    create_preview_clip(video_path, clip['start_time'], clip['end_time'], preview_output_path)
    
    return gr.update(value=preview_output_path, visible=True)

def run_clip_generation(video_path, clips_df, apply_reframe, progress=gr.Progress()):
    if not video_path or clips_df is None:
        raise gr.Error("Missing video path or clip data.")

    logging.info("Starting final clip generation...")
    output_files = []
    clips_to_generate = clips_df.to_dict('records')
    total_clips = len(clips_to_generate)
    temp_dir = "temp"
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, clip_data_row in enumerate(clips_to_generate):
        progress(i / total_clips, desc=f"Generating clip {i+1}/{total_clips}...")
        
        start_time = deformat_time(clip_data_row["Start"])
        end_time = deformat_time(clip_data_row["End"])
        filename = f"{clip_data_row['Filename']}.mp4"
        temp_clip_path = os.path.join(temp_dir, f"temp_clip_{i}.mp4")
        final_output_path = os.path.join(output_dir, filename)

        cut_clip_simple(video_path, start_time, end_time, temp_clip_path)

        if apply_reframe:
            reframe_video(temp_clip_path, final_output_path, aspect_ratio_str="9:16")
        else:
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            os.rename(temp_clip_path, final_output_path)
        
        if os.path.exists(final_output_path):
            output_files.append(final_output_path)

    progress(1, desc="All clips generated!")
    logging.info("Clip generation complete!")
    cleanup_temp_files()
    return f"Clips saved to the 'output' directory:\n" + "\n".join(output_files)

def run_batch_reframe(video_files, progress=gr.Progress()):
    if not video_files:
        raise gr.Error("Please upload at least one video file.")

    total_videos = len(video_files)
    output_files = []
    logging.info(f"Starting batch re-frame for {total_videos} videos.")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, video_file in enumerate(video_files):
        video_path = video_file.name
        progress(i / total_videos, desc=f"Re-framing video {i+1}/{total_videos}...")
        
        filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_reframed.mp4"
        output_path = os.path.join(output_dir, filename)
        
        reframe_video(video_path, output_path, aspect_ratio_str="9:16")
        if os.path.exists(output_path):
            output_files.append(output_path)

    progress(1, desc="Batch re-framing complete!")
    return f"Re-framed videos saved to the 'output' directory:\n" + "\n".join(output_files)

def run_chapter_generation_logic(input_type, youtube_url, local_file, whisper_model, language, llm_service, api_key, model_name, manual_model_name, min_chapters, max_chapters, progress=gr.Progress()):
    progress(0, desc="Starting Chapter Generation...")
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    if input_type == "YouTube URL" and youtube_url:
        video_path = download_video(youtube_url, temp_dir=temp_dir, progress=progress)
    elif input_type == "Local File" and local_file is not None:
        video_path = local_file.name
    else:
        raise gr.Error("Please provide a valid input for the selected type.")
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Failed to load video.")

    progress(0.2, desc="Extracting audio...")
    audio_path = extract_audio(video_path, temp_dir=temp_dir)
    progress(0.4, desc="Transcribing audio...")
    saved_settings = load_api_keys()
    device_setting = saved_settings.get("WHISPER_DEVICE", "auto").lower()
    device = "cuda" if device_setting == "gpu" else device_setting
    transcription_path = transcribe_audio(audio_path, model_name=whisper_model, language=language, device=device)
    with open(transcription_path, 'r', encoding='utf-8') as f:
        transcription_data = json.load(f)
    transcription_segments = transcription_data.get('segments', [])

    progress(0.6, desc="Getting video duration...")
    video_duration = get_video_duration(video_path)

    progress(0.7, desc="Generating chapters with LLM...")
    final_model_name = manual_model_name if manual_model_name and llm_service != "openrouter" else model_name
    if llm_service == "openrouter":
        final_model_name = manual_model_name

    chapter_generator = LLMChapterGenerator(api_key=api_key, model=final_model_name)
    chapter_suggestions = chapter_generator.generate_chapters(
        transcription_segments, video_duration, language=language,
        min_chapters=min_chapters, max_chapters=max_chapters
    )

    if not chapter_suggestions or "chapters" not in chapter_suggestions or not chapter_suggestions["chapters"]:
        raise gr.Error("No chapters were generated by the LLM.")

    progress(0.9, desc="Formatting chapters...")
    youtube_chapters = format_chapters_for_youtube(chapter_suggestions["chapters"])
    
    progress(1, desc="Chapter generation complete!")
    return youtube_chapters

def run_captioning_logic(input_type, youtube_url, local_file, whisper_model, language, bg_color, hl_color, txt_color, progress=gr.Progress()):
    progress(0, desc="Starting Captioning...")
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if input_type == "YouTube URL" and youtube_url:
        video_path = download_video(youtube_url, temp_dir=temp_dir, progress=progress)
    elif input_type == "Local File" and local_file is not None:
        video_path = local_file.name
    else:
        raise gr.Error("Please provide a valid input for the selected type.")
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Failed to load video.")

    progress(0.2, desc="Extracting audio...")
    audio_path = extract_audio(video_path, temp_dir=temp_dir)
    progress(0.4, desc="Transcribing audio for captions...")
    saved_settings = load_api_keys()
    device_setting = saved_settings.get("WHISPER_DEVICE", "auto").lower()
    device = "cuda" if device_setting == "gpu" else device_setting
    transcription_path = transcribe_audio(audio_path, model_name=whisper_model, language=language, device=device)
    with open(transcription_path, 'r', encoding='utf-8') as f:
        transcription_data = json.load(f)
    segments = transcription_data.get('segments', [])

    progress(0.6, desc="Processing segments...")
    import cv2
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    processed_segments = process_segments_for_captions(segments, width)

    progress(0.7, desc="Burning captions into video...")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_captioned.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    final_video_path = caption_video(video_path, output_path, processed_segments, bg_color, hl_color, txt_color)

    progress(1, desc="Captioning complete!")
    return final_video_path

def handle_save_api_keys(gemini, openai, claude, deepseek, openrouter, whisper_device):
    keys = {
        "GEMINI_API_KEY": gemini, "OPENAI_API_KEY": openai, "ANTHROPIC_API_KEY": claude,
        "DEEPSEEK_API_KEY": deepseek, "OPENROUTER_API_KEY": openrouter,
        "WHISPER_DEVICE": whisper_device
    }
    if save_api_keys(keys):
        gr.Info("API keys and paths saved successfully!")
    else:
        gr.Warning("Failed to save settings.")

def update_model_ui(service):
    service_config = models_config.get(service, {})
    model_choices = service_config.get("models", [])
    default_model = service_config.get("default", "")
    
    loaded_keys = load_api_keys()
    key_name = f"{service.upper()}_API_KEY"
    api_key = loaded_keys.get(key_name, "")
    
    is_openrouter = service == "openrouter"
    return (
        gr.update(visible=not is_openrouter, choices=model_choices, value=default_model),
        gr.update(visible=not is_openrouter),
        gr.update(visible=is_openrouter),
        gr.update(visible=is_openrouter),
        api_key
    )
