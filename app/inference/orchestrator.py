import hashlib, time
from datetime import datetime, timezone
from .utils import load_image

ANIMALS = ["dog","cat","bird","horse","lion"]
OBJECTS = ["soccer_ball","bottle","car","chair","laptop","phone"]
TEAMS   = ["Atlético Nacional","Millonarios","América de Cali","Junior","Santa Fe"]

def _hash_probs(b: bytes, classes: list[str]):
    h = hashlib.sha256(b).digest()
    vals = [h[i] / 255 for i in range(len(classes))]
    s = sum(vals) or 1.0
    probs = [v/s for v in vals]
    pairs = list(zip(classes, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

class Ensemble:
    def __init__(self):
        self.version = "yolo@1.0.0+logos@0.1.0+ocr@0.1.0"

    def health(self):
        return {"status":"ok","loaded":True,"model_version":self.version,
                "tasks":["objects","animals","teams","numbers"]}

    def run(self, image_bytes: bytes):
        t0 = time.time()
        img = load_image(image_bytes)  # valida
        objects = _hash_probs(image_bytes, OBJECTS)[:2]
        animals = _hash_probs(image_bytes, ANIMALS)[:2]
        teams   = _hash_probs(image_bytes, TEAMS)[:2]
        s = sum(image_bytes[:256])
        ocr = []
        if (s % 100) < 50:
            num = str((s % 90) + 10)
            ocr = [{"label": num, "score": 0.9}]
        summary = []
        if teams:   summary.append({"task":"team","label":teams[0][0], "score":round(teams[0][1],3)})
        if objects: summary.append({"task":"object","label":objects[0][0], "score":round(objects[0][1],3)})
        if animals: summary.append({"task":"animal","label":animals[0][0], "score":round(animals[0][1],3)})
        if ocr:     summary.append({"task":"number","label":ocr[0]["label"], "score":ocr[0]["score"]})
        detections = [{"label": lbl, "score": round(sc,3)} for lbl,sc in objects]
        topk = {
            "animals": [{"label": lbl, "score": round(sc,3)} for lbl,sc in animals],
            "teams":   [{"label": lbl, "score": round(sc,3)} for lbl,sc in teams],
        }
        return {
            "request_id": hashlib.md5(image_bytes).hexdigest(),
            "model_version": self.version,
            "summary": summary,
            "detections": detections,
            "ocr": [{"label": x["label"], "score": x["score"]} for x in ocr],
            "topk": topk,
            "timings_ms": {"total": round((time.time()-t0)*1000,2)},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
