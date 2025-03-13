import numpy as np
import pandas as pd
import json
import os
import sys
import traceback
from typing import Dict, Any, List, Union
import joblib
from datetime import datetime

from logger import get_logger, log_exception
from config import (
    GENETIC_DATA_THRESHOLD,
    MODEL_CACHE_DIR,
    DATA_DIR
)
from model import query_model

logger = get_logger('genetics')

class GeneticIntegration:
    def __init__(self):
        self.scaler = None
        self._load_components()
        logger.info("GeneticIntegration initialized successfully")
    
    def _load_components(self):
        try:
            model_dir = os.path.join(MODEL_CACHE_DIR, 'genetics')
            os.makedirs(model_dir, exist_ok=True)
            
            self.scaler_path = os.path.join(model_dir, 'genetic_scaler.joblib')
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded genetic data scaler")
        except Exception as e:
            log_exception(logger, e, "Error loading components")
            self.scaler = None
    
    def preprocess_genetic_markers(self, genetic_markers: List[Dict[str, Any]]) -> pd.DataFrame:
        try:
            if not genetic_markers:
                return pd.DataFrame()
                
            marker_dict = {}
            for marker in genetic_markers:
                marker_id = marker.get('id', f"unknown_{len(marker_dict)}")
                marker_value = marker.get('value', 0.0)
                marker_dict[marker_id] = marker_value
                
            df = pd.DataFrame([marker_dict])
            logger.info(f"Preprocessed genetic markers into DataFrame with shape {df.shape}")
            return df
        except Exception as e:
            log_exception(logger, e, "Error preprocessing genetic markers")
            return pd.DataFrame()
    
    def normalize_genetic_data(self, genetic_df: pd.DataFrame) -> np.ndarray:
        try:
            if genetic_df.empty:
                return np.array([])
                
            if genetic_df.isnull().values.any():
                genetic_df = genetic_df.fillna(genetic_df.mean())
            
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.fit(genetic_df)
                joblib.dump(self.scaler, self.scaler_path)
                
            processed_data = self.scaler.transform(genetic_df)
            return processed_data
        except Exception as e:
            log_exception(logger, e, "Error normalizing genetic data")
            return genetic_df.to_numpy()
    
    async def analyze_genetic_risk(self, genetic_markers: List[Dict[str, Any]], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Analyzing genetic risk for {len(genetic_markers)} markers")
            
            genetic_df = self.preprocess_genetic_markers(genetic_markers)
            
            if genetic_df.empty:
                return {
                    'risk_level': 'Unknown',
                    'confidence': 0.0,
                    'recommendations': [{
                        'type': 'warning',
                        'message': 'Insufficient genetic data provided for analysis.'
                    }]
                }
            
            normalized_data = self.normalize_genetic_data(genetic_df)
            
            # Prepare complete model input with all patient context
            model_input = {
                "genetic_markers": [
                    {"id": col, "value": float(normalized_data[0][i])}
                    for i, col in enumerate(genetic_df.columns)
                ],
                "patient_info": patient_info,
                "analysis_type": "genetic_risk"
            }
            
            # Let the model handle all analysis including family history, ethnicity, etc.
            model_response = await query_model(model_input, "genetic")
            
            # Add metadata
            model_response['timestamp'] = datetime.now().isoformat()
            model_response['data_points_analyzed'] = genetic_df.shape[0] * genetic_df.shape[1]
            
            return model_response
        except Exception as e:
            log_exception(logger, e, "Error analyzing genetic risk")
            return {
                'risk_level': 'Error',
                'confidence': 0.0,
                'recommendations': [{
                    'type': 'error',
                    'message': 'An error occurred during genetic risk analysis. Please consult with a healthcare professional.'
                }]
            }

async def get_personalized_recommendations(genetic_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        patient_id = genetic_data.get('patient_id', 'unknown')
        genetic_markers = genetic_data.get('genetic_markers', [])
        patient_info = genetic_data.get('patient_info', {})
        
        logger.info(f"Processing personalized recommendations for patient {patient_id}")
        
        integration = GeneticIntegration()
        recommendations = await integration.analyze_genetic_risk(genetic_markers, patient_info)
        
        response = {
            'patient_id': patient_id,
            'genetic_recommendations': recommendations,
            'processed_at': datetime.now().isoformat()
        }
        
        # Add data quality metrics
        data_completeness = min(1.0, len(genetic_markers) / 10.0)  # Assuming 10+ markers is complete data
        response['data_quality'] = {
            'completeness': data_completeness,
            'confidence_modifier': 0.5 + (data_completeness * 0.5)  # Scale from 0.5 to 1.0
        }
        
        return response
    except Exception as e:
        log_exception(logger, e, f"Error getting personalized recommendations: {str(e)}")
        
        return {
            'patient_id': genetic_data.get('patient_id', 'unknown'),
            'error': str(e),
            'status': 'failed',
            'processed_at': datetime.now().isoformat()
        }

if __name__ == "__main__":
    try:
        import asyncio
        input_data = json.loads(sys.stdin.read())
        result = asyncio.run(get_personalized_recommendations(input_data))
        print(json.dumps(result))
    except Exception as e:
        log_exception(logger, e, "Error in genetics.py main")
        print(json.dumps({
            'status': 'error',
            'error': str(e),
            'message': 'Failed to process genetic data',
            'timestamp': datetime.now().isoformat()
        }))
