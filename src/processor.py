import os
import subprocess
import torch

def cut_video_for_processing(video_path, start_time, end_time, output_path):
    """
    Cuts a video from start_time to end_time and re-encodes it.
    This is used to create a smaller clip for the main processing pipeline.
    """
    try:
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264', # Re-encode to ensure clean cut
            '-c:a', 'aac',
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except Exception as e:
        print(f"Error in cutting for processing: {e}")
        return None

def cut_clip_simple(video_path, start_time, end_time, output_path):
    """
    Cuts a video from start_time to end_time without re-encoding, using a direct FFmpeg call.
    """
    try:
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except Exception as e:
        print(f"Error in simple cut: {e}")
        return None

def create_preview_clip(video_path, start_time, end_time, output_path):
    """
    Quickly cuts a clip using FFmpeg, attempting to use GPU acceleration.
    """
    try:
        command = ['ffmpeg', '-y']
        if torch.cuda.is_available():
            command.extend(['-hwaccel', 'cuda'])
        command.extend(['-ss', str(start_time), '-to', str(end_time), '-i', video_path])
        if torch.cuda.is_available():
            command.extend(['-c:v', 'h264_nvenc'])
        else:
            command.extend(['-c:v', 'libx264'])
        command.extend(['-c:a', 'aac', output_path])
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except Exception as e:
        print(f"Error creating preview clip with GPU, falling back to CPU: {e}")
        try:
            command = ['ffmpeg', '-y', '-ss', str(start_time), '-to', str(end_time), '-i', video_path, '-c:v', 'libx264', '-c:a', 'aac', output_path]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except Exception as e2:
            print(f"CPU fallback for preview failed: {e2}")
            return None
