from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from fastapi.responses import FileResponse
import secrets

# Correct MFLUX import, straight from the docs
from mflux.models.flux.variants.txt2img.flux import Flux1

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    width: int
    height: int

@app.post("/generate")
def generate(req: GenerateRequest):
    seed = secrets.randbits(32)

    flux_model = Flux1.from_name(
        model_name="schnell",
        quantize=4,
    )

    output_dir = Path("/tmp/mflux-images")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"image_{ts}.png"
    output_path = output_dir / filename

    image = flux_model.generate_image(
        seed=seed,
        prompt=req.prompt,
        num_inference_steps=2,
        height=req.height,
        width=req.width,
    )
    image.save(path=str(output_path))

    image_url = f"http://localhost:8000/images/{filename}"

    return {
        "status": "ok",
        "prompt": req.prompt,
        "seed": seed,
        "image_path": str(output_path),
        "image_url": image_url,
    }

@app.get("/images/{filename}")
def get_image(filename: str):
    output_dir = Path("/tmp/mflux-images")
    file_path = output_dir / filename
    return FileResponse(str(file_path), media_type="image/png")