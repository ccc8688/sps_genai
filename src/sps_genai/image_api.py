from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision import transforms

from sps_genai.model import CNN

app = FastAPI()

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model = CNN()
model.load_state_dict(
    torch.load("cnn_cifar10.pth", map_location="cpu")
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


@app.get("/")
def read_root():
    return {"message": "CNN CIFAR10 Image Classification API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1).item()

    return {
        "class_id": predicted,
        "class_name": classes[predicted]
    }
