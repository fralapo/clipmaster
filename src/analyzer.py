import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import anthropic
import deepseek

load_dotenv()

class LLMClipFinder:
    def __init__(self, llm_service="gemini", api_key=None):
        self.llm_service = llm_service.lower()
        self.api_key = api_key
        self._configure_client()

    def _configure_client(self):
        if self.api_key:
            if self.llm_service == "gemini":
                genai.configure(api_key=self.api_key)
            elif self.llm_service == "openai" or self.llm_service == "openrouter":
                base_url = "https://openrouter.ai/api/v1" if self.llm_service == "openrouter" else None
                self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
            elif self.llm_service == "claude":
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif self.llm_service == "deepseek":
                self.client = deepseek.DeepSeek(api_key=self.api_key)
        else:
            # Fallback to environment variables
            if self.llm_service == "gemini":
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            elif self.llm_service == "openai":
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            elif self.llm_service == "openrouter":
                self.client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    def get_default_system_prompt(self, num_clips, duration_prompt, language="English"):
        prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        return prompt_template.format(
            num_clips=num_clips,
            duration_prompt=duration_prompt,
            language=language
        )

    def find_clips(self, transcription_text, context="", num_clips=5, duration="auto", model_name=None, language="English"):
        duration_map = {
            "30s": "exactly 30 seconds long", "30-60s": "between 30 and 60 seconds long",
            "60-90s": "between 60 and 90 seconds long", "auto": "between 30 and 90 seconds long"
        }
        duration_prompt = duration_map.get(duration, duration_map["auto"])
        system_prompt = self.get_default_system_prompt(num_clips, duration_prompt, language)
        user_prompt = f"Transcription:\n{transcription_text}"
        if context:
            user_prompt += f"\n\nAdditional Context from user:\n{context}"

        try:
            if self.llm_service == "gemini":
                model_to_use = model_name or 'gemini-1.5-pro-latest'
                generation_config = genai.GenerationConfig(response_mime_type="application/json")
                model = genai.GenerativeModel(model_to_use, generation_config=generation_config)
                response = model.generate_content([system_prompt, user_prompt])
                response_text = response.text
            else:
                model_to_use = model_name
                if not model_to_use:
                    if self.llm_service == "openai": model_to_use = "gpt-4o"
                    elif self.llm_service == "claude": model_to_use = "claude-3-5-sonnet-20240620"
                    elif self.llm_service == "deepseek": model_to_use = "deepseek-chat"
                    elif self.llm_service == "openrouter": model_to_use = "openrouter/auto"
                
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                
                if self.llm_service in ["openai", "deepseek", "openrouter"]:
                    completion = self.client.chat.completions.create(
                        model=model_to_use, messages=messages, response_format={"type": "json_object"}
                    )
                    response_text = completion.choices[0].message.content
                elif self.llm_service == "claude":
                    completion = self.client.messages.create(
                        model=model_to_use, max_tokens=4096, messages=messages
                    )
                    response_text = completion.content[0].text
            
            response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(response_text)

        except Exception as e:
            print(f"Error interacting with {self.llm_service} using model {model_name}: {e}")
            return None

def analyze_transcription(transcription_path, context="", num_clips=5, llm_service="gemini", api_key=None, duration="auto", model_name=None, language="English"):
    if not os.path.exists(transcription_path):
        print(f"Error: Transcription file not found at {transcription_path}")
        return None

    with open(transcription_path, 'r', encoding='utf-8') as f:
        transcription_data = json.load(f)
    
    transcription_text = transcription_data['text']

    finder = LLMClipFinder(llm_service=llm_service, api_key=api_key)
    clip_suggestions = finder.find_clips(transcription_text, context, num_clips, duration, model_name, language)

    if clip_suggestions:
        output_path = os.path.join(os.path.dirname(transcription_path), "clip_suggestions.json")
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(clip_suggestions, f, indent=4, ensure_ascii=False)
        print(f"Clip suggestions saved to: {output_path}")
        return output_path
    else:
        print("Failed to get clip suggestions from the LLM.")
        return None
