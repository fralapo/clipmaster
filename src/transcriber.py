import os
import json
import torch
import whisper

def transcribe_audio(audio_path, model_name="base", language=None, device="auto"):
    """
    Transcribes an audio file using the specified Whisper model,
    generating word-level timestamps.
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return None

    if device == "auto" or device is None:
        processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        processing_device = device
    
    print(f"Using device: {processing_device}")

    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name, device=processing_device)

        print(f"Starting transcription for {audio_path}...")
        transcribe_options = {"word_timestamps": True}
        if language and language.lower() != 'auto':
            transcribe_options['language'] = language
        
        result = model.transcribe(audio_path, **transcribe_options)

        output_path = os.path.join(os.path.dirname(audio_path), "transcription.json")
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"Transcription successful. Output saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
