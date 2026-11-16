import io
import os
import base64
import torch
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from torchvision import transforms

from sps_genai.model import CNN
from sps_genai.generative_models import EnergyModel, DiffusionModel

app = FastAPI(title="SPS GenAI CIFAR-10 Classification & Generation API")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model = CNN()
if os.path.exists("cnn_cifar10.pth"):
    model.load_state_dict(torch.load("cnn_cifar10.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

ebm_model = EnergyModel().to(DEVICE)
if os.path.exists("ebm_cifar10.pth"):
    ebm_model.load_state_dict(torch.load("ebm_cifar10.pth", map_location=DEVICE))
ebm_model.eval()

diffusion_model = DiffusionModel(timesteps=200).to(DEVICE)
if os.path.exists("diffusion_cifar10.pth"):
    diffusion_model.load_state_dict(torch.load("diffusion_cifar10.pth", map_location=DEVICE))
diffusion_model.eval()

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    img = (tensor.clamp(-1.0, 1.0) + 1.0) * 0.5
    img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img, mode="RGB")

@app.get("/")
def read_root():
    return {"message": "SPS GenAI API is running!", "endpoints": ["/predict", "/generate/ebm", "/generate/diffusion"]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        predicted = torch.argmax(model(image), dim=1).item()
    return {"class_id": predicted, "class_name": classes[predicted]}

@app.post("/generate/ebm")
@app.get("/generate/ebm")
def generate_ebm(
    steps: int = Query(60, ge=5, le=200),
    step_size: float = Query(10.0, gt=0.0),
    as_base64: bool = Query(False),
):
    sample_tensor = ebm_model.sample(batch_size=1, steps=steps, step_size=step_size, device=DEVICE)[0]
    buf = io.BytesIO()
    tensor_to_pil(sample_tensor).save(buf, format="PNG")
    buf.seek(0)
    if as_base64:
        return JSONResponse({"model": "ebm", "resolution": "32x32", "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")})
    return StreamingResponse(buf, media_type="image/png")

@app.post("/generate/diffusion")
@app.get("/generate/diffusion")
def generate_diffusion(as_base64: bool = Query(False)):
    sample_tensor = diffusion_model.sample(batch_size=1, device=DEVICE)[0]
    buf = io.BytesIO()
    tensor_to_pil(sample_tensor).save(buf, format="PNG")
    buf.seek(0)
    if as_base64:
        return JSONResponse({"model": "diffusion", "resolution": "32x32", "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")})
    return StreamingResponse(buf, media_type="image/png")
