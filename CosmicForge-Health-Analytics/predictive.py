import pandas as pd
import numpy as np
import json
import os
import sys
import traceback
from typing import Dict, Any, List, Union
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from logger import get_logger, log_exception
from config import (
    PREDICTION_CONFIDENCE_THRESHOLD,
    CRITICAL_ALERT_THRESHOLD,
    MODEL_CACHE_DIR,
    DATA_DIR
)
from model import query_model

logger = get_logger('predictive')

class PredictiveAnalytics:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self._load_components()
        logger.info("PredictiveAnalytics initialized successfully")

    def _load_components(self):
        try:
            model_dir = os.path.join(MODEL_CACHE_DIR, 'predictive')
            os.makedirs(model_dir, exist_ok=True)
            
            self.scaler_path = os.path.join(model_dir, 'predictive_scaler.joblib')
            self.imputer_path = os.path.join(model_dir, 'predictive_imputer.joblib')
            
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded predictive data scaler")
                
            if os.path.exists(self.imputer_path):
                self.imputer = joblib.load(self.imputer_path)
                logger.info("Loaded predictive data imputer")
                
        except Exception as e:
            log_exception(logger, e, "Error loading components")
            self.scaler = None
            self.imputer = None
    
    def preprocess_health_data(self, health_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info(f"Preprocessing health data with shape {health_data.shape}")
            
            if health_data.empty:
                return {"processed_data": np.array([]), "original_data": {}}
            
            # Create a copy of the original data for the model
            original_data = health_data.copy().to_dict('records')[0]
            
            # Handle non-numeric values in health metrics
            for col in health_data.columns:
                if col == 'blood_pressure' and col in health_data:
                    try:
                        # Extract systolic and diastolic from format like "120/80"
                        bp_values = health_data[col].str.split('/', expand=True)
                        health_data['systolic'] = pd.to_numeric(bp_values[0], errors='coerce')
                        health_data['diastolic'] = pd.to_numeric(bp_values[1], errors='coerce')
                        health_data = health_data.drop(columns=[col])
                    except Exception as e:
                        log_exception(logger, e, f"Error processing blood pressure values: {str(e)}")
                        health_data = health_data.drop(columns=[col])
            
            # Convert all columns to numeric, coercing errors to NaN
            for col in health_data.columns:
                health_data[col] = pd.to_numeric(health_data[col], errors='coerce')
            
            # Initialize imputer if not already done
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')
                imputed_data = self.imputer.fit_transform(health_data)
                joblib.dump(self.imputer, self.imputer_path)
            else:
                imputed_data = self.imputer.transform(health_data)
            
            # Initialize scaler if not already done
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(imputed_data)
                joblib.dump(self.scaler, self.scaler_path)
            else:
                scaled_data = self.scaler.transform(imputed_data)
            
            return {
                "processed_data": scaled_data,
                "original_data": original_data,
                "column_names": health_data.columns.tolist()
            }
            
        except Exception as e:
            log_exception(logger, e, "Error preprocessing health data")
            return {
                "processed_data": np.array([]),
                "original_data": health_data.to_dict('records')[0] if not health_data.empty else {}
            }
    
    async def analyze_patient_health(self, health_data: pd.DataFrame, patient_info: Dict[str, Any], health_history: pd.DataFrame = None) -> Dict[str, Any]:
        try:
            logger.info(f"Analyzing patient health with {health_data.shape[1]} metrics")
            
            if health_data.empty:
                return {
                    "status": "Warning",
                    "message": "No health metrics provided",
                    "recommendations": [{
                        "type": "warning",
                        "message": "Unable to analyze health without metrics data."
                    }]
                }
            
            # Preprocess health data
            preprocessing_result = self.preprocess_health_data(health_data)
            
            if preprocessing_result["processed_data"].size == 0:
                return {
                    "status": "Error",
                    "message": "Failed to process health metrics",
                    "recommendations": [{
                        "type": "warning",
                        "message": "Unable to process health metrics data. Please check the format."
                    }]
                }
            
            # Prepare input for model
            model_input = {
                "health_metrics": preprocessing_result["original_data"],
                "patient_info": patient_info,
                "analysis_type": "health_prediction"
            }
            
            # Add processed data for advanced analysis
            model_input["processed_metrics"] = {
                "data": preprocessing_result["processed_data"].tolist(),
                "columns": preprocessing_result.get("column_names", [])
            }
            
            # Add health history if available
            if health_history is not None and not health_history.empty:
                model_input["health_history"] = health_history.to_dict('records')
            
            # Query model
            model_response = await query_model(model_input, "predictive")
            
            # Add metadata
            model_response['timestamp'] = datetime.now().isoformat()
            model_response['data_points_analyzed'] = health_data.shape[0] * health_data.shape[1]
            
            return model_response
            
        except Exception as e:
            log_exception(logger, e, "Error analyzing patient health")
            return {
                'status': 'Error',
                'error': str(e),
                'recommendations': [
                    {
                        'type': 'error',
                        'message': 'An error occurred during health analysis. Please consult with a healthcare professional.'
                    }
                ]
            }

async def analyze_patient_health(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        patient_id = patient_data.get('patient_id', 'unknown')
        health_metrics = patient_data.get('health_metrics', {})
        patient_info = patient_data.get('patient_info', {})
        health_history = patient_data.get('health_history', [])
        
        logger.info(f"Analyzing health for patient {patient_id}")
        
        # Convert health metrics to DataFrame
        if isinstance(health_metrics, dict):
            health_df = pd.DataFrame([health_metrics])
        elif isinstance(health_metrics, list) and health_metrics:
            health_df = pd.DataFrame(health_metrics)
        else:
            health_df = pd.DataFrame()
        
        # Convert health history to DataFrame
        if isinstance(health_history, list) and health_history:
            history_df = pd.DataFrame(health_history)
        else:
            history_df = pd.DataFrame()
        
        # Initialize PredictiveAnalytics
        analytics = PredictiveAnalytics()
        
        # Get health analysis from model
        analysis_result = await analytics.analyze_patient_health(health_df, patient_info, history_df)
        
        # Prepare final result
        result = {
            'patient_id': patient_id,
            'analysis': analysis_result,
            'processed_at': datetime.now().isoformat()
        }
        
        # Add alert if critical issues detected
        if analysis_result.get('requires_attention', False):
            result['alert'] = {
                'level': analysis_result.get('alert_level', 'normal'),
                'message': analysis_result.get('alert_message', 'Potential health issue detected.'),
                'timestamp': datetime.now().isoformat()
            }
        
        return result
        
    except Exception as e:
        log_exception(logger, e, f"Error analyzing patient health: {str(e)}")
        
        return {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'error': str(e),
            'status': 'failed',
            'processed_at': datetime.now().isoformat(),
            'recommendations': [{
                'type': 'error',
                'message': 'System error occurred during health analysis. Please try again or contact support.'
            }]
        }

if __name__ == "__main__":
    try:
        import asyncio
        input_data = json.loads(sys.stdin.read())
        result = asyncio.run(analyze_patient_health(input_data))
        print(json.dumps(result))
    except Exception as e:
        log_exception(logger, e, "Error in predictive.py main")
        print(json.dumps({
            'status': 'error',
            'error': str(e),
            'message': 'Failed to process health data',
            'timestamp': datetime.now().isoformat()
        }))
