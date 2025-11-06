import json, logging
logger = logging.getLogger("telemetry")
logging.basicConfig(level=logging.INFO)
_counters = {"requests":0,"errors":0}
def log_prediction(payload): 
    _counters["requests"] += 1; logger.info(json.dumps({"type":"prediction", **payload})[:2000])
def log_error(msg:str):
    _counters["errors"] += 1; logger.error(msg)
def get_metrics(): return _counters
