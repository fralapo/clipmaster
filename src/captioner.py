import os
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import List, Dict, Any, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

def cv2_to_pil(cv2_img):
    """Convert CV2 image (BGR) to PIL image (RGB)"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_img):
    """Convert PIL image (RGB) to CV2 image (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_rounded_rectangle(draw, bbox, radius, fill):
    """Draw a rounded rectangle"""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius <= 0:
        draw.rectangle((x1, y1, x2, y2), fill=fill)
        return
    draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
    draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)
    draw.pieslice((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=fill)
    draw.pieslice((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=fill)
    draw.pieslice((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=fill)
    draw.pieslice((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=fill)

def parse_color(color_str: str) -> Tuple[int, int, int, int]:
    """Parse color string from Gradio ColorPicker (hex or rgba)."""
    import re
    color_str = color_str.strip()
    
    # Handle hex format (#RRGGBB or #RRGGBBAA)
    if color_str.startswith('#'):
        hex_color = color_str.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return r, g, b, 255
        elif len(hex_color) == 8:
            r, g, b, a = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
            return r, g, b, a

    # Handle rgba format (e.g., "rgba(255.123, 10, 20, 0.5)")
    match = re.match(r'rgba?\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)(?:,\s*([\d.]+))?\)', color_str)
    if match:
        r, g, b, a_str = match.groups()
        r, g, b = int(float(r)), int(float(g)), int(float(b))
        a = int(float(a_str) * 255) if a_str is not None else 255
        return r, g, b, a

    raise ValueError(f"Invalid color format: {color_str}")

def process_segments_for_captions(segments, video_width):
    """Process segments to create text lines for captions"""
    font_path = "clipmaster/src/fonts/Poppins-Regular.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf" # Fallback
    font_size = max(28, int(video_width * 0.03))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        logger.warning(f"Font not found at {font_path}, using default.")
        font = ImageFont.load_default()

    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    
    # Use textlength for more accurate width calculation
    char_width = draw.textlength("a", font=font)
    usable_width = int(video_width * 0.8)
    chars_per_line = int(usable_width / char_width) if char_width > 0 else 30

    for segment in segments:
        words_in_segment = segment.get("words", [])
        if not words_in_segment:
            continue

        text_lines = []
        current_line_words = []
        line_start_time = words_in_segment[0]['start']

        for i, word_info in enumerate(words_in_segment):
            current_line_words.append(word_info)
            line_text = " ".join(w['word'] for w in current_line_words)

            # Break line if it's too long or if there's a significant pause
            pause_threshold = 1.0 # seconds
            is_last_word = (i == len(words_in_segment) - 1)
            next_word_start = words_in_segment[i+1]['start'] if not is_last_word else float('inf')
            pause_after_word = next_word_start - word_info['end']

            if len(line_text) > chars_per_line or pause_after_word > pause_threshold or is_last_word:
                text_lines.append({
                    "text": line_text,
                    "words": current_line_words,
                    "start": line_start_time,
                    "end": word_info["end"]
                })
                current_line_words = []
                if not is_last_word:
                    line_start_time = next_word_start
        
        segment["text_lines"] = text_lines
    
    return segments

def caption_video(video_path, output_path, segments, bg_color_str, highlight_color_str, text_color_str):
    """Add captions to the entire video with animated word highlights."""
    try:
        bg_color = parse_color(bg_color_str)
        highlight_color = parse_color(highlight_color_str)
        text_color = parse_color(text_color_str)[:3]
    except ValueError as e:
        logger.error(f"Invalid color format: {e}")
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_output_path = f"{output_path}_temp.mp4"
    # Use 'mp4v' as it is more broadly compatible than 'avc1' (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error("Could not open video writer with mp4v codec. Check your OpenCV/FFmpeg installation.")
        return None
    
    font_path = "clipmaster/src/fonts/Poppins-Regular.ttf"
    if not os.path.exists(font_path):
        font_path = "arial.ttf" # Fallback
    font_size = max(28, int(width * 0.03))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    bottom_padding = int(height * 0.05)
    last_text_end_time = 0
    silence_duration = 2.0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        
        active_lines = []
        for segment in segments:
            for line in segment.get("text_lines", []):
                if line["start"] <= current_time < line["end"]:
                    active_lines.append(line)
        
        show_caption = bool(active_lines)
        if not show_caption:
            if current_time - last_text_end_time < silence_duration:
                # Find the most recent line to keep displaying it during the pause
                recent_lines = [line for s in segments for line in s.get("text_lines", []) if line["end"] <= current_time]
                if recent_lines:
                    most_recent = max(recent_lines, key=lambda x: x["end"])
                    if current_time - most_recent["end"] < silence_duration:
                        active_lines = [most_recent]
                        show_caption = True

        if show_caption and active_lines:
            pil_img = cv2_to_pil(frame)
            draw = ImageDraw.Draw(pil_img, 'RGBA')

            for i, line in enumerate(active_lines[:2]):
                line_text = line["text"]
                bbox = draw.textbbox((0, 0), line_text, font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                center_x = width // 2
                text_y = height - bottom_padding - (text_height * (len(active_lines) - i)) - (10 * (len(active_lines) - 1 - i))

                bg_padding = int(font_size * 0.4)
                bg_bbox = (center_x - text_width/2 - bg_padding, text_y - bg_padding, 
                           center_x + text_width/2 + bg_padding, text_y + text_height + bg_padding)
                draw_rounded_rectangle(draw, bg_bbox, int(font_size * 0.5), fill=bg_color)

                current_x = center_x - text_width / 2
                for word_info in line.get("words", []):
                    word_text = word_info["word"]
                    word_bbox = draw.textbbox((0,0), word_text, font=font)
                    word_width = word_bbox[2] - word_bbox[0]
                    
                    if word_info["start"] <= current_time < word_info["end"]:
                        highlight_bbox = (current_x, text_y, current_x + word_width, text_y + text_height)
                        draw_rounded_rectangle(draw, highlight_bbox, int(font_size * 0.2), fill=highlight_color)

                    draw.text((current_x, text_y), word_text, font=font, fill=text_color)
                    space_width = draw.textlength(" ", font=font)
                    current_x += word_width + space_width
            
            frame = pil_to_cv2(pil_img)
            last_text_end_time = max(line["end"] for line in active_lines)

        out.write(frame)

    cap.release()
    out.release()
    
    final_output = output_path
    combine_cmd = f"ffmpeg -i {temp_output_path} -i {video_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest {final_output} -y"
    try:
        subprocess.run(combine_cmd, shell=True, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio muxing failed: {e.stderr.decode()}")
        os.rename(temp_output_path, final_output) # Save video without audio
        return final_output

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    return final_output if os.path.exists(final_output) else None
