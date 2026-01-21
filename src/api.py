from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import io
import os
from src.data_ingestion import UniversalParser
from src.data_fusion import DataFusion
from src.model import SpectralCNN
from src.discovery import ChemicalDiscovery
from src.utils import FAMILIES
import shutil

app = FastAPI(title="Meteorite Organics Spectral Analysis API")

# Load model (global)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "best_model.pth"
model = SpectralCNN(num_classes=len(FAMILIES))
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

parser = UniversalParser()
fusion = DataFusion()
discovery = ChemicalDiscovery(model, DEVICE)

@app.post("/predict")
async def predict_spectrum(file: UploadFile = File(...)):
    # Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Parse based on extension or heuristic
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".tab":
            spectrum = parser.parse_pds_tab(temp_path)
        else:
            # Generic parser
            spectrum = parser.parse_pds_tab(temp_path) # Fallback for now

        if not spectrum:
            return {"error": "Could not parse spectral file."}
        
        # 2. Process & Broaden/Align
        _, y_proc, _ = fusion.process_and_merge([spectrum])
        x_input = torch.FloatTensor(y_proc).to(DEVICE)
        
        # 3. Model Inference
        with torch.no_grad():
            logits, embs = model(x_input)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(1)
        
        # 4. Saliency/Explainability
        saliency = discovery.generate_saliency_map(y_proc, target_class=pred_idx.item())
        
        # 5. Extract top peaks from saliency
        top_peaks_idx = np.argsort(saliency)[-5:] # Top 5 peaks
        top_peaks_wavenumbers = fusion.common_x[top_peaks_idx].tolist()
        
        return {
            "id": spectrum.id,
            "prediction": FAMILIES[pred_idx.item()],
            "confidence": float(conf.item()),
            "embedding": embs.squeeze().cpu().numpy().tolist(),
            "top_peaks_cm1": top_peaks_wavenumbers,
            "saliency_map": saliency.tolist()
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
