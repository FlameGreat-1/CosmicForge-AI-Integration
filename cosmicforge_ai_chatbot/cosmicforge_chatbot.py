import json
import uuid
import datetime
import os
from cachetools import LRUCache
from .config import Config
from .logger import setup_logger
import asyncio
from typing import Tuple, Optional
from fastapi import HTTPException
from .model import CosmicForgeAIChatbot

logger = setup_logger()

class CosmicForgeMedicalChat:
    def __init__(self):
        self.model = CosmicForgeAIChatbot()
        self.response_cache = LRUCache(maxsize=100)

    async def initialize(self):
        await self.model.load_model()

    def create_medical_chat_prompt(self, user_input: str) -> str:
        return f"""You are CosmicForge, an advanced medical AI assistant. Provide concise, professional responses to medical queries including:
1. Key Information
2. Brief Explanation
3. Recommendations

User Input: {user_input}

If the input is not related to medical or health topics, respond ONLY with: "I'm CosmicForge, a medical AI assistant. I can only provide information on health and medical topics."
"""

    async def process_chat(self, user_input: str) -> Tuple[str, str]:
        logger.info(f"Processing user input: {user_input}")
        try:
            prompt = self.create_medical_chat_prompt(user_input)
            chat_result = await self._get_model_response(prompt)
            clean_response = await self._post_process_response(chat_result)

            response_id = await self.save_response(clean_response)
            logger.info(f"Generated response: {clean_response}")
            logger.info(f"Response ID: {response_id}")        

            return clean_response, response_id
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to process chat request")

    async def _get_model_response(self, prompt: str) -> str:
        if prompt in self.response_cache:
            return self.response_cache[prompt]

        response = await self.model.generate_response(prompt)
        clean_response = await self._remove_prompt(prompt, response)
        self.response_cache[prompt] = clean_response
        return clean_response

    async def _remove_prompt(self, prompt: str, response: str) -> str:
        clean_response = response.replace(prompt, "").strip()
        lines = clean_response.split('\n')
        clean_lines = [line for line in lines if not line.startswith("User Input:") and "CosmicForge" not in line]
        final_response = '\n'.join(clean_lines).strip()
        return final_response

    async def _post_process_response(self, response: str) -> str:
        response = response.replace('Response:', '').replace('"', '').replace("'", "").strip()
        first_section = "Key Information:"
        if first_section in response:
            response = response[response.index(first_section):]
        sections = ["Key Information:", "Brief Explanation:", "Recommendations:"]
        clean_lines = []
        current_section = ""
        for line in response.split('\n'):
            line = line.strip()
            if any(section in line for section in sections):
                if clean_lines:
                    clean_lines.append(" ")  # Single space between sections
                current_section = line.strip(':')
                clean_lines.append(current_section + ":")
            elif line:
                if current_section == "Key Information":
                    clean_lines.append(line.strip('* '))
                elif current_section == "Recommendations":
                    clean_lines.append("- " + line.lstrip('0123456789. '))
                else:
                    clean_lines.append(line)

        clean_output = " ".join(clean_lines)

        if "I'm CosmicForge, a medical AI assistant." in clean_output:
            return clean_output
        
        clean_output += " Important note: This information is for educational purposes only and does not constitute medical advice. Please consult a healthcare professional for personalized medical guidance."

        return clean_output

    async def save_response(self, chat_result: str) -> str:
        response_id = str(uuid.uuid4())
        file_path = os.path.join(Config.DATA_DIR, f"chat_response_{response_id}.json")

        async with asyncio.Lock():
            try:
                with open(file_path, mode='w') as file:
                    json.dump({
                        "response": chat_result,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "status": "completed"
                    }, file, indent=2)
            except IOError as e:
                logger.error(f"Error saving chat response: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to save chat response")

        logger.info(f"Chat response saved to {file_path}")
        return response_id

    async def get_response(self, response_id: str) -> Optional[dict]:
        file_path = os.path.join(Config.DATA_DIR, f"chat_response_{response_id}.json")

        try:
            async with asyncio.Lock():
                with open(file_path, mode='r') as file:
                    response_data = json.load(file)
            return response_data
        except FileNotFoundError:
            logger.error(f"Chat response with ID {response_id} not found")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error decoding chat response data for ID {response_id}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving chat response: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve chat response")

    async def get_response_status(self, response_id: str) -> str:
        file_path = os.path.join(Config.DATA_DIR, f"chat_response_{response_id}.json")
        if os.path.exists(file_path):
            try:
                async with asyncio.Lock():
                    with open(file_path, mode='r') as file:
                        response_data = json.load(file)
                return response_data.get("status", "completed")
            except json.JSONDecodeError:
                logger.error(f"Error decoding chat response data for ID {response_id}")
                return "error"
            except Exception as e:
                logger.error(f"Unexpected error retrieving chat response status: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to retrieve chat response status")
        return "not found"