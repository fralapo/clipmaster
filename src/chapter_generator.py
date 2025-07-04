import os
import subprocess
import numpy as np
import json
import time
import google.generativeai as genai
from typing import List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

class LLMChapterGenerator:
    """Class to handle LLM API calls for identifying YouTube chapters"""

    def __init__(self, api_key=None, model="gemini-1.5-flash"):
        """Initialize with optional API key (for Google Gemini)"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model

        if not self.api_key:
            logger.warning("No Google Gemini API key found for chapter generation.")
            self.use_gemini = False
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.use_gemini = True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API for chapters: {e}")
            self.use_gemini = False

    def generate_chapters(self, transcription_segments, video_duration, language="English", min_chapters=5, max_chapters=15):
        """Use LLM to identify logical chapter points from transcription segments"""
        
        transcript_text = ""
        for i, segment in enumerate(transcription_segments):
            start_time = self._format_time(segment["start"])
            transcript_text += f"[{start_time}] {segment['text']}\n"
        
        duration_minutes = video_duration / 60
        
        prompt = f"""
You are a professional YouTube video editor who specializes in creating effective chapter markers.
The video and transcript are in {language}.

Below is a transcript of a video that is {duration_minutes:.1f} minutes long.
The transcript includes timestamps in [hh:mm:ss] format.

TRANSCRIPT:
{transcript_text}

Please generate {min_chapters}-{max_chapters} logical chapter markers for this video by dividing it into meaningful sections.

REQUIREMENTS:
1. The first chapter MUST start at 00:00.
2. Chapters should represent natural topical divisions or key moments.
3. Choose concise but descriptive chapter titles (3-7 words each). The titles MUST be in {language}.
4. Ensure time gaps between chapters aren't too short (at least 10+ seconds).
5. Space chapters somewhat evenly throughout the video, but prioritize content transitions.

Format your response as a single JSON object with this exact structure:
{{
  "chapters": [
    {{
      "time": "00:00",
      "title": "Introduction in {language}"
    }},
    {{
      "time": "01:23",
      "title": "First Main Point in {language}"
    }},
    ...
  ]
}}

Ensure ALL timestamps are in the MM:SS or HH:MM:SS format required by YouTube.
"""

        if self.use_gemini:
            return self._call_gemini_api(prompt)
        else:
            logger.warning("Gemini not available, using fallback chapter generation.")
            return self._fallback_extraction(transcription_segments, video_duration)
            
    def _call_gemini_api(self, prompt):
        """Call Gemini API with proper error handling"""
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            try:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)
                
                chapter_data = json.loads(content)
                return chapter_data
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from LLM response. Using manual extraction.")
                return self._manually_extract_chapters(content)
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return None
            
    def _manually_extract_chapters(self, content):
        """Manually extract chapter information if JSON parsing fails"""
        chapters = []
        chapter_matches = re.findall(r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:-|:|–|\|)?\s*([^\n]+)', content)
        
        for time_str, title in chapter_matches:
            chapters.append({
                "time": time_str,
                "title": title.strip()
            })
        
        return {"chapters": chapters}
    
    def _fallback_extraction(self, transcription_segments, video_duration):
        """Simple fallback method if API calls fail"""
        chapters = []
        chapters.append({"time": "00:00", "title": "Introduction"})
        
        num_chapters = 5
        segment_duration = video_duration / num_chapters
        
        for i in range(1, num_chapters):
            chapter_time = i * segment_duration
            closest_segment = min(transcription_segments, key=lambda x: abs(x["start"] - chapter_time))
            
            if closest_segment:
                title_text = closest_segment["text"]
                if len(title_text) > 40:
                    title_text = title_text[:37] + "..."
                
                chapters.append({
                    "time": self._format_time(closest_segment["start"]),
                    "title": title_text
                })
        
        return {"chapters": chapters}
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS or HH:MM:SS format for YouTube"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

def get_video_duration(video_path: str) -> float:
    """Get the duration of the video in seconds using ffprobe."""
    try:
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
        return float(output)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed to get duration for {video_path}: {e.output.decode('utf-8')}")
        return 0.0
    except Exception as e:
        logger.error(f"Error getting video duration for {video_path}: {e}")
        return 0.0

def time_to_seconds(time_str: str) -> int:
    """Convert time string (MM:SS or HH:MM:SS) to seconds for sorting"""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        return 0

def format_chapters_for_youtube(chapters: List[Dict[str, str]]) -> str:
    """Format chapters in YouTube-ready format"""
    if not chapters:
        return "No chapters generated."
    
    has_zero_start = any(chapter["time"] == "00:00" for chapter in chapters)
    
    if not has_zero_start:
        chapters.insert(0, {"time": "00:00", "title": "Introduction"})
    
    chapters.sort(key=lambda x: time_to_seconds(x["time"]))
    
    youtube_format = ""
    for chapter in chapters:
        youtube_format += f"{chapter['time']} {chapter['title']}\n"
    
    return youtube_format.strip()
