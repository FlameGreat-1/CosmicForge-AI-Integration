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
from .model import MedicalDiagnosisModel

logger = setup_logger()

class MedicalDiagnosis:
    def __init__(self):
        self.model = MedicalDiagnosisModel()
        self.response_cache = LRUCache(maxsize=100)

    async def initialize(self):
        await self.model.load_model()

    def create_diagnosis_prompt(self, user_input: str) -> str:
        return f"""You are a medical AI assistant designed ONLY for providing diagnoses based on medical information. You must NEVER engage in any task or conversation unrelated to medical diagnosis. If asked about anything else, politely refuse and state that you can only provide medical diagnoses.

User Input: {user_input}

Provide a concise, professional medical diagnosis including:
1. Potential condition(s)
2. Brief explanation
3. Recommendations

If the input is not related to medical symptoms, conditions, or patient information, respond ONLY with: "I'm sorry, but I can only provide medical diagnoses based on medical information. I cannot assist with other topics or tasks."
"""

    async def diagnose(self, user_input: str) -> Tuple[str, str]:
        logger.info(f"Processing user input: {user_input}")
        try:
            prompt = self.create_diagnosis_prompt(user_input)
            diagnosis_result = await self._get_model_response(prompt)
            clean_diagnosis = await self._post_process_diagnosis(diagnosis_result)

            diagnosis_id = await self.save_diagnosis(clean_diagnosis)
            logger.info(f"Generated diagnosis: {clean_diagnosis}")
            logger.info(f"Diagnosis ID: {diagnosis_id}")        

            return clean_diagnosis, diagnosis_id
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to process diagnosis request")

    async def _get_model_response(self, prompt: str) -> str:
        if prompt in self.response_cache:
            return self.response_cache[prompt]

        response = await self.model.generate_diagnosis(prompt)
        clean_response = await self._remove_prompt(prompt, response)
        self.response_cache[prompt] = clean_response
        return clean_response

    async def _remove_prompt(self, prompt: str, response: str) -> str:
        clean_response = response.replace(prompt, "").strip()
        lines = clean_response.split('\n')
        clean_lines = [line for line in lines if not line.startswith("Please provide") and "diagnosis" not in line.lower()]
        final_response = ' '.join(clean_lines).strip()
        return final_response

    async def _post_process_diagnosis(self, diagnosis: str) -> str:
        diagnosis = diagnosis.replace('Diagnosis Result:', '').replace('"', '').replace("'", "").strip()
        first_section = "Potential condition(s):"
        if first_section in diagnosis:
            diagnosis = diagnosis[diagnosis.index(first_section):]
        sections = ["Potential condition(s):", "Brief explanation:", "Recommendations:"]
        clean_lines = []
        current_section = ""
        for line in diagnosis.split('\n'):
            line = line.strip()
            if any(section in line for section in sections):
                if clean_lines:
                    clean_lines.append(" ")  # Single space between sections
                current_section = line.strip(':')
                clean_lines.append(current_section + ":")
            elif line:
                if current_section == "Potential condition(s)":
                    clean_lines.append(line.strip('* '))
                elif current_section == "Recommendations":
                    clean_lines.append("- " + line.lstrip('0123456789. '))
                else:
                    clean_lines.append(line)

        clean_output = " ".join(clean_lines)  
        
        return clean_output

    async def save_diagnosis(self, diagnosis_result: str) -> str:
        diagnosis_id = str(uuid.uuid4())
        file_path = os.path.join(Config.DATA_DIR, f"diagnosis_{diagnosis_id}.json")

        async with asyncio.Lock():
            try:
                with open(file_path, mode='w') as file:
                    json.dump({
                        "diagnosis": diagnosis_result,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "status": "completed"
                    }, file, indent=2)
            except IOError as e:
                logger.error(f"Error saving diagnosis: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to save diagnosis")

        logger.info(f"Diagnosis saved to {file_path}")
        return diagnosis_id

    async def get_diagnosis(self, diagnosis_id: str) -> Optional[dict]:
        file_path = os.path.join(Config.DATA_DIR, f"diagnosis_{diagnosis_id}.json")

        try:
            async with asyncio.Lock():
                with open(file_path, mode='r') as file:
                    diagnosis_data = json.load(file)
            return diagnosis_data
        except FileNotFoundError:
            logger.error(f"Diagnosis with ID {diagnosis_id} not found")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error decoding diagnosis data for ID {diagnosis_id}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving diagnosis: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve diagnosis")

    async def get_diagnosis_status(self, diagnosis_id: str) -> str:
        file_path = os.path.join(Config.DATA_DIR, f"diagnosis_{diagnosis_id}.json")
        if os.path.exists(file_path):
            try:
                async with asyncio.Lock():
                    with open(file_path, mode='r') as file:
                        diagnosis_data = json.load(file)
                return diagnosis_data.get("status", "completed")
            except json.JSONDecodeError:
                logger.error(f"Error decoding diagnosis data for ID {diagnosis_id}")
                return "error"
            except Exception as e:
                logger.error(f"Unexpected error retrieving diagnosis status: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to retrieve diagnosis status")
        return "not found"
