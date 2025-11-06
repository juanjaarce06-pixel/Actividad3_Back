from PIL import Image
from io import BytesIO
def load_image(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGB")
