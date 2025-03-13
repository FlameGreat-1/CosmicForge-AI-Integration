import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import os
import asyncio
from fastapi import HTTPException
import psutil
from typing import Dict, List, Union, Optional, Any
from datetime import datetime
import json
from logger import get_logger, log_exception
from config import MODEL_CACHE_DIR, DATA_DIR, MODEL_PATH, API_HOST, API_PORT

logger = get_logger('model')

def split_model_into_shards(model_path, shard_size=1000000000):
    logger.info(f"Checking if model sharding is needed for: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise ValueError(f"Model path {model_path} does not exist. Only local models are supported.")

    # Create a model-specific directory for shards
    model_shard_dir = os.path.join(DATA_DIR, "model_shards")
    os.makedirs(model_shard_dir, exist_ok=True)

    try:
        logger.info(f"Loading model from local path: {model_path}")
        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            local_files_only=True,  
            low_cpu_mem_usage=True
        )
        
        state_dict = model.state_dict()
        
        total_model_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        available_memory = psutil.virtual_memory().available
        
        logger.info(f"Model size: {total_model_size / 1e9:.2f} GB, Available memory: {available_memory / 1e9:.2f} GB")
        
        if available_memory > total_model_size * 1.2:
            logger.info("Sufficient memory available, skipping sharding")
            return model_path
            
        current_shard = {}
        current_shard_size = 0
        shard_index = 0

        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current_shard_size + tensor_size > shard_size:
                shard_path = os.path.join(model_shard_dir, f"model_shard_{shard_index}.pt")
                torch.save(current_shard, shard_path)
                logger.info(f"Saved shard {shard_index} to {shard_path}")

                current_shard = {}
                current_shard_size = 0
                shard_index += 1

            current_shard[key] = tensor
            current_shard_size += tensor_size

        if current_shard:
            shard_path = os.path.join(model_shard_dir, f"model_shard_{shard_index}.pt")
            torch.save(current_shard, shard_path)
            logger.info(f"Saved shard {shard_index} to {shard_path}")

        logger.info(f"Model split into {shard_index + 1} shards")
        
        del model
        del state_dict
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return model_shard_dir  
        
    except Exception as e:
        logger.error(f"Error splitting model into shards: {str(e)}", exc_info=True)
        raise  


class MemoryEfficientShardedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.shard_size = 1000000000  # 1GB shard size, adjust as needed
        self.loaded_shards = {}
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing_enable()

    def load_shard(self, shard_id):
        model_shard_dir = os.path.join(DATA_DIR, "model_shards")
        shard_path = os.path.join(model_shard_dir, f"model_shard_{shard_id}.pt")
        if os.path.exists(shard_path):
            self.loaded_shards[shard_id] = torch.load(shard_path, map_location='cpu')
            self.load_state_dict(self.loaded_shards[shard_id], strict=False)
        else:
            logger.warning(f"Shard {shard_id} not found")

    def unload_shard(self, shard_id):
        if shard_id in self.loaded_shards:
            del self.loaded_shards[shard_id]
            torch.cuda.empty_cache()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        required_shards = set(input_ids.div(self.shard_size, rounding_mode='floor').unique().tolist())
        
        for shard_id in required_shards:
            if shard_id not in self.loaded_shards:
                self.load_shard(shard_id)
        
        for shard_id in list(self.loaded_shards.keys()):
            if shard_id not in required_shards:
                self.unload_shard(shard_id)
        
        return super().forward(input_ids, attention_mask, **kwargs)

    def parallelize(self):
        self.model_parallel = True
        self.device_map = "auto"
        self.deparallelize()
        self.parallelize()


class ModelInterface:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelInterface, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            logger.info(f"ModelInterface initialized")
        return cls._instance

    async def load_model(self):
        logger.info("Loading model")
        try:
            model_path = MODEL_PATH
            persistent_path = os.path.join(DATA_DIR, "health_analytics_model")
            os.makedirs(persistent_path, exist_ok=True)
            
            # Create model shards directory
            model_shard_dir = os.path.join(DATA_DIR, "model_shards")
            os.makedirs(model_shard_dir, exist_ok=True)

            # Check if shards exist, if not, create them
            if not any(file.startswith("model_shard_") for file in os.listdir(model_shard_dir)):
                logger.info("Model shards not found. Creating shards...")
                await asyncio.to_thread(split_model_into_shards, model_path)
                logger.info("Model shards created successfully")

            logger.info(f"Loading tokenizer from {model_path}")
            self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_path, local_files_only=True)
            
            logger.info(f"Tokenizer type: {type(self.tokenizer)}")
            logger.info(f"Tokenizer class: {self.tokenizer.__class__.__name__}")
            logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")

            # Check for GPU availability
            if torch.cuda.is_available():
                logger.info("GPU is available. Using 8-bit quantization.")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf4",
                    bnb_8bit_compute_dtype=torch.float16
                )
                quantization_config = bnb_config
            else:
                logger.info("GPU is not available. Loading model in 16-bit precision.")
                quantization_config = None
            config = LlamaForCausalLM.config_class.from_pretrained(model_path, local_files_only=True)
            self.model = await asyncio.to_thread(
                MemoryEfficientShardedLlamaForCausalLM.from_pretrained,
                model_path,
                config=config,
                local_files_only=True,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )

            # Use model parallelism if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                self.model.parallelize()

            logger.info("Model loaded successfully")
            logger.info(f"Current memory usage: {psutil.virtual_memory().percent}%")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load the model")
                                    
    async def query_model(self, prompt: str) -> Dict[str, Any]:
        logger.info("Querying model")
        try:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            inputs = await asyncio.to_thread(
                self.tokenizer, 
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            start_time.record()
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=300,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000  
            
            generated_text = await asyncio.to_thread(self.tokenizer.decode, outputs[0], skip_special_tokens=True)
            logger.info(f"Model query completed in {elapsed_time:.2f}s")
            
            # Clean up memory after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            return {"generated_text": generated_text}
        except Exception as e:
            logger.error(f"Error querying model: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to query model")

    def format_prompt(self, data: Dict[str, Any], prompt_type: str) -> str:
        if prompt_type == "genetic":
            return self._format_genetic_prompt(data)
        elif prompt_type == "predictive":
            return self._format_predictive_prompt(data)
        elif prompt_type == "lifestyle":
            return self._format_lifestyle_prompt(data)
        else:
            return self._format_general_prompt(data)

    def _format_genetic_prompt(self, data: Dict[str, Any]) -> str:
        patient_info = data.get('patient_info', {})
        genetic_markers = data.get('genetic_markers', [])
        
        marker_text = ""
        for i, marker in enumerate(genetic_markers[:20]):
            if isinstance(marker, dict):
                marker_id = marker.get('id', f'Marker_{i}')
                marker_value = marker.get('value', 'Unknown')
                marker_text += f"- {marker_id}: {marker_value}\n"
            else:
                marker_text += f"- Marker_{i}: {marker}\n"
        
        prompt = f"[INST] Analyze these genetic markers for patient age {patient_info.get('age', 'Unknown')}, gender {patient_info.get('gender', 'Unknown')}, ethnicity {patient_info.get('ethnicity', 'Unknown')}. Family history: {', '.join(patient_info.get('family_history', []))}. Markers: {marker_text} Provide risk assessment, preventative measures, lifestyle recommendations, and genetic counseling needs. [/INST]"
        return prompt.strip()

    def _format_predictive_prompt(self, data: Dict[str, Any]) -> str:
        patient_info = data.get('patient_info', {})
        health_metrics = data.get('health_metrics', {})
        
        metrics_text = ", ".join([f"{k}: {v}" for k, v in health_metrics.items()])
        
        prompt = f"[INST] Analyze health data for patient age {patient_info.get('age', 'Unknown')}, gender {patient_info.get('gender', 'Unknown')}, blood group {patient_info.get('blood_group', 'Unknown')}. Health metrics: {metrics_text}. Symptoms: {', '.join(patient_info.get('symptoms', []))}. Medical history: {', '.join(patient_info.get('medical_history', []))}. Identify potential health issues, risk levels, recommended tests, and preventative measures. [/INST]"
        return prompt.strip()

    def _format_lifestyle_prompt(self, data: Dict[str, Any]) -> str:
        patient_info = data.get('patient_info', {})
        lifestyle_data = data.get('lifestyle_data', {})
        
        prompt = f"[INST] Analyze lifestyle for patient age {patient_info.get('age', 'Unknown')}, gender {patient_info.get('gender', 'Unknown')}. Lifestyle factors: {json.dumps(lifestyle_data)}. Medication adherence: {data.get('adherence_score', 'Unknown')}. Analyze medication effectiveness, adherence strategies, lifestyle modifications, and adherence barriers. [/INST]"
        return prompt.strip()

    def _format_general_prompt(self, data: Dict[str, Any]) -> str:
        query_type = data.get('analysis_type', 'health assessment')
        patient_info = data.get('patient_info', {})
        
        prompt = f"[INST] Provide {query_type} for patient age {patient_info.get('age', 'Unknown')}, gender {patient_info.get('gender', 'Unknown')}. Include relevant medical information, personalized recommendations, warning signs, and when to consult healthcare professionals. [/INST]"
        return prompt.strip()

    def process_response(self, response: Any) -> Dict[str, Any]:
        try:
            if isinstance(response, dict):
                generated_text = response.get('generated_text', '')
            else:
                generated_text = str(response)
            
            structured_response = self._extract_structured_data(generated_text)
            structured_response['timestamp'] = datetime.now().isoformat()
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
                        
            return structured_response
        except Exception as e:
            log_exception(logger, e, "Error processing response")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
                      
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        result = {'recommendations': [], 'risk_factors': [], 'summary': ''}
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                item = line.lstrip('- 0123456789. ')
                if any(risk_word in item.lower() for risk_word in ['risk', 'condition', 'disease', 'disorder']):
                    result['risk_factors'].append(item)
                else:
                    result['recommendations'].append({'recommendation': item})
        
        sentences = text.split('.')
        if sentences:
            result['summary'] = '.'.join(sentences[:2]) + '.'
            
        return result

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.load_model()
        return cls._instance


async def query_model(data: Dict[str, Any], prompt_type: str = "general") -> Dict[str, Any]:
    try:
        model_interface = await ModelInterface.get_instance()
        prompt = model_interface.format_prompt(data, prompt_type)
        response = await model_interface.query_model(prompt)

        # Store the result in a variable
        result = model_interface.process_response(response)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result
    except Exception as e:
        logger.error(f"Error in query_model function: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")