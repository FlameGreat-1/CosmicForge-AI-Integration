import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, Any, List
import joblib
from datetime import datetime

from logger import get_logger, log_exception
from config import MODEL_CACHE_DIR
from model import query_model

logger = get_logger('lifestyle')

class LifestyleInsights:
    def __init__(self):
        self.scaler = None
        self._load_components()
        logger.info("LifestyleInsights initialized")

    def _load_components(self):
        try:
            model_dir = os.path.join(MODEL_CACHE_DIR, 'lifestyle')
            os.makedirs(model_dir, exist_ok=True)
            self.scaler_path = os.path.join(model_dir, 'lifestyle_scaler.joblib')
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            log_exception(logger, e, "Error loading components")
            self.scaler = None

    async def analyze_medication_adherence(self, prescription_data: List[Dict], patient_reports: List[str]) -> Dict[str, Any]:
        try:
            if not prescription_data or not patient_reports:
                return {"adherence_score": 0, "status": "Insufficient data"}
            
            model_input = {
                "prescription_data": prescription_data,
                "patient_reports": patient_reports,
                "analysis_type": "medication_adherence"
            }
            
            response = await query_model(model_input, "lifestyle")
            return response
        except Exception as e:
            log_exception(logger, e, "Error analyzing medication adherence")
            return {"adherence_score": 0, "status": "Error", "error": str(e)}

    async def extract_symptom_insights(self, patient_reports: List[str]) -> List[Dict[str, Any]]:
        try:
            if not patient_reports:
                return []
            
            model_input = {
                "patient_reports": patient_reports,
                "analysis_type": "symptom_extraction"
            }
            
            response = await query_model(model_input, "lifestyle")
            return response.get("symptom_insights", [])
        except Exception as e:
            log_exception(logger, e, "Error extracting symptom insights")
            return []

    async def analyze_lifestyle_impact(self, lifestyle_data: Dict[str, Any], medication_effectiveness: List[float]) -> List[Dict[str, Any]]:
        try:
            if not lifestyle_data:
                return []
                
            model_input = {
                "lifestyle_data": lifestyle_data,
                "medication_effectiveness": medication_effectiveness,
                "analysis_type": "lifestyle_impact"
            }
            
            response = await query_model(model_input, "lifestyle")
            return response.get("lifestyle_factors", [])
        except Exception as e:
            log_exception(logger, e, "Error analyzing lifestyle impact")
            return []

async def get_lifestyle_insights(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        patient_id = patient_data.get('patient_id', 'unknown')
        prescription_data = patient_data.get('prescription_data', [])
        patient_reports = patient_data.get('patient_reports', [])
        lifestyle_data = patient_data.get('lifestyle_data', {})
        medication_effectiveness = patient_data.get('medication_effectiveness', [])
        patient_info = patient_data.get('patient_info', {})
        
        insights = LifestyleInsights()
        
        adherence_results = await insights.analyze_medication_adherence(prescription_data, patient_reports)
        symptom_insights = await insights.extract_symptom_insights(patient_reports)
        lifestyle_impact = await insights.analyze_lifestyle_impact(lifestyle_data, medication_effectiveness)
        
        model_input = {
            "patient_id": patient_id,
            "adherence_results": adherence_results,
            "symptom_insights": symptom_insights,
            "lifestyle_impact": lifestyle_impact,
            "patient_info": patient_info,
            "analysis_type": "comprehensive_insights"
        }
        
        try:
            model_response = await query_model(model_input, "lifestyle")
        except Exception as e:
            log_exception(logger, e, "Error getting model recommendations")
            model_response = {"error": str(e)}
        
        result = {
            'patient_id': patient_id,
            'adherence_score': adherence_results.get('adherence_score', 0),
            'adherence_status': adherence_results.get('status', 'Unknown'),
            'symptom_insights': symptom_insights,
            'lifestyle_impact': lifestyle_impact,
            'recommendations': model_response.get('recommendations', []),
            'processed_at': datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        log_exception(logger, e, "Error getting lifestyle insights")
        return {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'error': str(e),
            'status': 'failed',
            'processed_at': datetime.now().isoformat()
        }

if __name__ == "__main__":
    try:
        import asyncio
        input_data = json.loads(sys.stdin.read())
        result = asyncio.run(get_lifestyle_insights(input_data))
        print(json.dumps(result))
    except Exception as e:
        log_exception(logger, e, "Error in lifestyle.py main")
        print(json.dumps({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }))
