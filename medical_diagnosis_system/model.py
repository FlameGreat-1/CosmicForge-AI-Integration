import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from .config import Config
from .logger import setup_logger
import transformers
import asyncio
import os
from fastapi import HTTPException
import psutil

logger = setup_logger()

def split_model_into_shards(model_path, shard_size=1000000000):
    logger.info(f"Checking if model sharding is needed for: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise ValueError(f"Model path {model_path} does not exist. Only local models are supported.")
    
    # Create a model-specific directory for shards
    model_shard_dir = os.path.join(Config.DATA_DIR, "model_shards")
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
        model_shard_dir = os.path.join(Config.DATA_DIR, "model_shards")
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

class MedicalDiagnosisModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedicalDiagnosisModel, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Transformers version: {transformers.__version__}")
        return cls._instance

    async def load_model(self):
        logger.info("Loading MedicalDiagnosis model")
        try:
            model_path = Config.MODEL_PATH
            persistent_path = os.path.join(Config.DATA_DIR, "medical_diagnosis_model")
            os.makedirs(persistent_path, exist_ok=True)
            
            # Create model shards directory
            model_shard_dir = os.path.join(Config.DATA_DIR, "model_shards")
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
                logger.info("GPU is not available. Loading model in 32-bit precision.")
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
            )

            # Use model parallelism if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                self.model.parallelize()

            logger.info("MedicalDiagnosis model loaded successfully")
            logger.info(f"Current memory usage: {psutil.virtual_memory().percent}%")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load the medical diagnosis model")

    async def generate_diagnosis(self, prompt: str) -> str:
        logger.info("Generating diagnosis")
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
    
            logger.info("Diagnosis generated successfully")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating diagnosis: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate diagnosis")

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.load_model()
        return cls._instance
