from pydantic import BaseModel
import os
class Settings(BaseModel):
    model_bucket: str = os.getenv("MODEL_BUCKET", "")
    model_manifest: str = os.getenv("MODEL_MANIFEST", "")
    google_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    allow_origins: list[str] = ["*"]
settings = Settings()
