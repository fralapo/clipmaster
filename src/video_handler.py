import os
import yt_dlp
from moviepy.editor import VideoFileClip

def download_video(url, temp_dir="temp", progress=None):
    """
    Downloads a video from a YouTube URL to the specified directory.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    def progress_hook(d):
        if d['status'] == 'downloading' and progress:
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total_bytes:
                percentage = d['downloaded_bytes'] / total_bytes
                progress(percentage, desc=f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'progress_hooks': [progress_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_dict)
            print(f"Video downloaded successfully to: {video_path}")
            return video_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def extract_audio(video_path, temp_dir="temp", progress=None):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    audio_path = os.path.join(temp_dir, "temp_audio.wav")

    try:
        if progress:
            progress(0.8, desc="Extracting audio...")
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        video_clip.close()
        if progress:
            progress(1, desc="Audio extracted.")
        print(f"Audio extracted successfully to: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_audio_segment(video_path, start_time, end_time, output_path, progress=None):
    """
    Extracts a specific segment of audio from a video file.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    try:
        if progress:
            progress(0, desc="Extracting audio segment...")
        
        with VideoFileClip(video_path) as video_clip:
            audio_segment = video_clip.subclip(start_time, end_time).audio
            audio_segment.write_audiofile(output_path, codec='pcm_s16le', logger=None)
        
        if progress:
            progress(1, desc="Audio segment extracted.")
        
        print(f"Audio segment extracted successfully to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error extracting audio segment: {e}")
        return None
