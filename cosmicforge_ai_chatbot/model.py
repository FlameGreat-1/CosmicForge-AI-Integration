import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from .config import Config
from .logger import setup_logger
import transformers
import asyncio
import os
from fastapi import HTTPException
import psutil
import json

logger = setup_logger()

def split_model_into_shards(model_path, shard_size=1000000000):
    logger.info(f"Checking if model sharding is needed for: {model_path}")
    
    model_shard_dir = os.path.join(Config.DATA_DIR, "model_shards")
    os.makedirs(model_shard_dir, exist_ok=True)
    
    # Check if model is already sharded
    if any(file.startswith("model_shard_") for file in os.listdir(model_shard_dir)):
        logger.info("Model is already sharded. Skipping sharding process.")
        return model_shard_dir

    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise ValueError(f"Model path {model_path} does not exist. Only local models are supported.")
    
    try:
        logger.info(f"Loading model from local path: {model_path}")
        with torch.no_grad():
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = cls.load_config(pretrained_model_name_or_path)
        model = cls(config)
        model.load_pretrained_shards(pretrained_model_name_or_path)
        return model

    @staticmethod
    def load_config(model_path):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Fix rope_scaling if it's present and not in the correct format
        if "rope_scaling" in config_dict:
            rope_scaling = config_dict["rope_scaling"]
            if isinstance(rope_scaling, dict):
                if "type" not in rope_scaling or "factor" not in rope_scaling:
                    config_dict["rope_scaling"] = {
                        "type": "linear",
                        "factor": rope_scaling.get("factor", 1.0)
                    }
            else:
                config_dict["rope_scaling"] = {
                    "type": "linear",
                    "factor": 1.0
                }
        
        # Remove any extra fields that are not part of the standard configuration
        standard_fields = ["type", "factor"]
        if "rope_scaling" in config_dict and isinstance(config_dict["rope_scaling"], dict):
            config_dict["rope_scaling"] = {k: v for k, v in config_dict["rope_scaling"].items() if k in standard_fields}

            # Ensure other necessary parameters are present
            config_dict.setdefault("rope_theta", 10000)
            config_dict.setdefault("rope_scaling_factor", 1.0)
            
            return LlamaForCausalLM.config_class.from_dict(config_dict)

    def load_pretrained_shards(self, model_path):
        model_shard_dir = os.path.join(Config.DATA_DIR, "model_shards")
        for shard_file in os.listdir(model_shard_dir):
            if shard_file.startswith("model_shard_") and shard_file.endswith(".pt"):
                shard_id = int(shard_file.split("_")[-1].split(".")[0])
                self.load_shard(shard_id)

    def load_shard(self, shard_id):
        try:
            model_shard_dir = os.path.join(Config.DATA_DIR, "model_shards")
            shard_path = os.path.join(model_shard_dir, f"model_shard_{shard_id}.pt")
            if os.path.exists(shard_path):
                self.loaded_shards[shard_id] = torch.load(shard_path, map_location='cpu')
                self.load_state_dict(self.loaded_shards[shard_id], strict=False)
            else:
                logger.warning(f"Shard {shard_id} not found")
        except Exception as e:
            logger.error(f"Error loading shard {shard_id}: {str(e)}", exc_info=True)

    def unload_shard(self, shard_id):
        try:
            if shard_id in self.loaded_shards:
                del self.loaded_shards[shard_id]
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error unloading shard {shard_id}: {str(e)}", exc_info=True)

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

class CosmicForgeAIChatbot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CosmicForgeAIChatbot, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Transformers version: {transformers.__version__}")
        return cls._instance

    async def load_model(self):
        if self.model is not None:
            logger.info("Model is already loaded.")
            return

        logger.info("Loading CosmicForge AI Chatbot model")
        try:
            model_path = Config.MODEL_PATH
            persistent_path = os.path.join(Config.DATA_DIR, "cosmicforge_ai_chatbot_model")
            os.makedirs(persistent_path, exist_ok=True)
            
            model_shard_dir = os.path.join(Config.DATA_DIR, "model_shards")
            os.makedirs(model_shard_dir, exist_ok=True)

            if not any(file.startswith("model_shard_") for file in os.listdir(model_shard_dir)):
                logger.info("Model shards not found. Creating shards...")
                await asyncio.to_thread(split_model_into_shards, model_path)
                logger.info("Model shards created successfully")

            logger.info(f"Loading tokenizer from {model_path}")
            self.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_path, local_files_only=True)
            
            logger.info(f"Tokenizer type: {type(self.tokenizer)}")
            logger.info(f"Tokenizer class: {self.tokenizer.__class__.__name__}")
            logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")

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
                logger.info("GPU is not available. Loading model in 32-bit precision.")
                quantization_config = None

            config = MemoryEfficientShardedLlamaForCausalLM.load_config(model_path)
            logger.info(f"Model config: {config}")  # This will log the configuration, including rope_scaling

            self.model = await asyncio.to_thread(
                MemoryEfficientShardedLlamaForCausalLM.from_pretrained,
                model_path,
                config=config,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            if torch.cuda.device_count() > 1:
                self.model.parallelize()

            logger.info("CosmicForge AI Chatbot model loaded successfully")
            logger.info(f"Current memory usage: {psutil.virtual_memory().percent}%")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load the CosmicForge AI Chatbot model")

    async def generate_response(self, prompt: str) -> str:
        logger.info("Generating response")
        try:
            inputs = await asyncio.to_thread(
                self.tokenizer, 
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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
            
            generated_text = await asyncio.to_thread(self.tokenizer.decode, outputs[0], skip_special_tokens=True)
          
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            logger.info("Response generated successfully")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate response")

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.load_model()
        return cls._instance

    async def cleanup(self):
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        logger.info("Model cleanup completed")

