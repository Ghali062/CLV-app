
try:
    import gdown
except ImportError:
    print("❌ gdown non trouvé. Vérifie que gdown est bien dans requirements.txt")
    raise
import os

MODEL_PATH = "clv_model_pipeline.pkl"
GDRIVE_ID = "1McG3vrjueVBdl5daB_2vCAEjcSY6amVU" # ID du fichier

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("Model already exists.")
