from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

DEVICE = torch.device("cpu")
IMG_SIZE = 224

# Load model
print("🔄 Loading model...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
try:
    model.load_state_dict(torch.load('model_pytorch.pth', map_location=DEVICE))
    model.to(DEVICE).eval()
    print("✅ Model loaded!")
except:
    print("❌ Run train_pytorch.py first!")
    model = None

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, 0)
        
        return {
            "prediction": "LEAF" if pred.item() == 1 else "NOT A LEAF",
            "confidence": round(conf.item() * 100, 2)
        }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    print("🚀 API: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)