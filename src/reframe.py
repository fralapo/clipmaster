"""
Core video reframing logic.
This module handles resizing a video to fit a new aspect ratio with padding.
"""
import os
import cv2
import numpy as np
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _get_ffmpeg_path():
    """Finds the path to the ffmpeg executable."""
    return shutil.which('ffmpeg')

def _mux_video_audio_with_ffmpeg(
    original_video_path: str,
    processed_video_path: str,
    final_output_path: str,
    ffmpeg_exec_path: str
) -> bool:
    """Combines the processed video with audio from the original using FFmpeg."""
    cmd = [
        ffmpeg_exec_path, '-y',
        '-i', processed_video_path,
        '-i', original_video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        final_output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')
        if result.returncode == 0:
            return True
        else:
            logger.error(f"FFmpeg failed (exit code {result.returncode}):\n{result.stderr[:500]}")
            return False
    except Exception as e:
        logger.error(f"Exception during FFmpeg execution: {e}", exc_info=True)
        return False

def _apply_fit_padding(base_frame: np.ndarray, content_to_fit: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scales content to fit inside the base_frame, preserving aspect ratio."""
    output_frame = base_frame.copy()
    content_h, content_w = content_to_fit.shape[:2]
    if content_h == 0 or content_w == 0:
        return output_frame

    scale = min(target_h / content_h, target_w / content_w)
    scaled_w, scaled_h = int(content_w * scale), int(content_h * scale)

    if scaled_w == 0 or scaled_h == 0:
        return output_frame

    resized_content = cv2.resize(content_to_fit, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    pad_x = (target_w - scaled_w) // 2
    pad_y = (target_h - scaled_h) // 2

    output_frame[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = resized_content
    return output_frame

# --- Main Reframe Function ---

def reframe_video(
    input_path: str,
    output_path: str,
    aspect_ratio_str: str = "9:16",
    output_height: int = 1080
) -> Optional[str]:
    """
    Re-frames a video to fit a target aspect ratio by adding padding.

    Args:
        input_path: Path to the input video file.
        output_path: Path to save the reframed output video.
        aspect_ratio_str: Target aspect ratio (e.g., "9:16").
        output_height: Target height for the output video.

    Returns:
        The path to the output file if successful, otherwise None.
    """
    try:
        w_str, h_str = aspect_ratio_str.split(':')
        aspect_ratio = float(w_str) / float(h_str)
    except ValueError:
        logger.error(f"Invalid aspect ratio format: {aspect_ratio_str}. Should be 'W:H'.")
        return None

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {input_path}")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_h = output_height
    out_w = int(round(out_h * aspect_ratio))
    if out_w % 2 != 0:
        out_w += 1

    temp_video_path = Path(output_path).parent / f"{Path(output_path).stem}_temp_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (out_w, out_h))

    if not out_writer.isOpened():
        logger.error(f"Could not open video writer for: {temp_video_path}")
        cap.release()
        return None

    pbar = tqdm(total=total_frames, desc="Reframing")

    logger.info("Applying 'Fit (Add Padding)' reframe.")
    black_base = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        final_frame = _apply_fit_padding(black_base, frame, out_w, out_h)
        out_writer.write(final_frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()

    logger.info("Muxing audio...")
    ffmpeg_path = _get_ffmpeg_path()
    if ffmpeg_path:
        success = _mux_video_audio_with_ffmpeg(input_path, str(temp_video_path), output_path, ffmpeg_path)
        if success:
            logger.info(f"Successfully created reframed video with audio: {output_path}")
            os.remove(temp_video_path)
            return output_path
        else:
            logger.warning("Muxing failed. The output video will be silent.")
            shutil.move(str(temp_video_path), output_path)
            return output_path
    else:
        logger.warning("FFmpeg not found. The output video will be silent.")
        shutil.move(str(temp_video_path), output_path)
        return output_path
