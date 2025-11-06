from pydantic import BaseModel
from typing import List, Dict, Optional

class BBox(BaseModel):
    x: float; y: float; w: float; h: float

class Detection(BaseModel):
    label: str; score: float; bbox: Optional[BBox] = None

class PredictSummary(BaseModel):
    task: str; label: str; score: float

class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    summary: List[PredictSummary]
    detections: List[Detection] = []
    ocr: List[Detection] = []
    topk: Dict[str, List[Detection]] = {}
    timings_ms: Dict[str, float] = {}
    timestamp: str
