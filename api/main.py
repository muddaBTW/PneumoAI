from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import os
import asyncio
from api.schemas import PredcitionResponse, ChatRequest, ChatResponse
from api.medical_chat import get_medical_chat_response

# Reduce noisy TensorFlow/absl startup logs until the model is actually loaded
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "healthy", "message": "PneumoAI API is running"}

# loading our model
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
model_path = root_dir / 'model' / 'pneumonia_resnet_model.keras'

# Lazy-loaded model and associated preprocess function
model = None
_preprocess_input = None
# Protect concurrent loads
_model_lock = asyncio.Lock()


async def _ensure_model_loaded():
    global model, _preprocess_input
    if model is not None:
        return
    async with _model_lock:
        if model is not None:
            return
        # Import TensorFlow lazily to avoid heavy startup at import time
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import preprocess_input as _pi
        # Load model in a thread to avoid blocking the event loop
        loaded = await asyncio.to_thread(tf.keras.models.load_model, str(model_path))
        model = loaded
        _preprocess_input = _pi


# Optional: preload model at application startup to avoid first-request cold start
@app.on_event("startup")
async def _maybe_preload_model_on_startup():
    import os
    preload = os.getenv("PRELOAD_MODEL", "true").lower()
    if preload in ("1", "true", "yes"):
        # schedule preload but don't block startup
        try:
            asyncio.create_task(_ensure_model_loaded())
        except Exception:
            # best-effort: ignore failures here, first request will load model instead
            pass


# Health / warm-up endpoint that forces model load when called
@app.get("/warmup")
async def warmup():
    try:
        await _ensure_model_loaded()
        return {"status": "ok", "model_loaded": model is not None}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# creating our endpoint
@app.post('/predict',response_model = PredcitionResponse)
async def predict(file:UploadFile = File(...)):
    # read the uploaded files in bytes
    images_bytes = await file.read()

    # convert bytes to PIL then to RGB
    img = Image.open(io.BytesIO(images_bytes))  # Open image first
    img = img.convert("RGB")                  # Then convert

    # set to input size
    img = img.resize((224,224))

    # convert to array
    img_array = np.array(img)

    # add batch dim
    img_array = np.expand_dims(img_array,axis = 0)

    # ensure model is loaded and get preprocess function
    await _ensure_model_loaded()
    img_array = _preprocess_input(img_array)

    # run prediction in threadpool to avoid blocking
    pred_arr = await asyncio.to_thread(model.predict, img_array)
    pred = float(pred_arr[0][0])

    # decide label and the confidence
    if pred > 0.5:
        prediction = 'Pneumonia'
        confidence = pred*100
    else:
        prediction = 'Normal'
        confidence = (1-pred)*100

    # return structured response
    return PredcitionResponse(
        prediction=prediction,
        probability=pred,
        confidence=confidence
    )

@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    response_text = get_medical_chat_response(
        message=request.message,
        prediction=request.prediction_context,
        confidence=request.confidence_context,
        image_b64=request.image_b64,
        model_id=request.model_id
    )
    return ChatResponse(response=response_text)