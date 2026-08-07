from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import os
import asyncio
from api.schemas import PredcitionResponse, ChatRequest, ChatResponse
from api.medical_chat import get_medical_chat_response

# loading our model
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
model_path = root_dir / 'model' / 'pneumonia_resnet_model.tflite'

# Lazy-loaded TFLite interpreter
_interpreter = None
_input_details = None
_output_details = None
# Protect concurrent loads
_model_lock = asyncio.Lock()


def _resnet_preprocess(img_array: np.ndarray) -> np.ndarray:
    """ResNet50 caffe-style preprocessing (replaces tf.keras preprocess_input).

    Converts RGB to BGR and subtracts ImageNet channel means.
    """
    x = img_array.astype(np.float32)
    # RGB -> BGR
    x = x[..., ::-1]
    # Subtract ImageNet mean values (BGR order)
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


async def _ensure_model_loaded():
    global _interpreter, _input_details, _output_details
    if _interpreter is not None:
        return
    async with _model_lock:
        if _interpreter is not None:
            return

        def _load():
            from tflite_runtime.interpreter import Interpreter
            interp = Interpreter(model_path=str(model_path))
            interp.allocate_tensors()
            return interp

        # Load model in a thread to avoid blocking the event loop
        interp = await asyncio.to_thread(_load)
        _input_details = interp.get_input_details()
        _output_details = interp.get_output_details()
        _interpreter = interp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler: preload model in background after server starts."""
    preload = os.getenv("PRELOAD_MODEL", "true").lower()
    if preload in ("1", "true", "yes"):
        # Schedule model preload as a background task so port binds immediately
        asyncio.create_task(_ensure_model_loaded())
    yield
    # Shutdown: nothing to clean up


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    """Health check endpoint — supports both GET and HEAD for Render."""
    body = {"status": "healthy", "message": "PneumoAI API is running"}
    return JSONResponse(content=body)


# Health / warm-up endpoint that forces model load when called
@app.get("/warmup")
async def warmup():
    try:
        await _ensure_model_loaded()
        return {"status": "ok", "model_loaded": _interpreter is not None}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# creating our endpoint
@app.post('/predict', response_model=PredcitionResponse)
async def predict(file: UploadFile = File(...)):
    # read the uploaded files in bytes
    images_bytes = await file.read()

    # convert bytes to PIL then to RGB
    img = Image.open(io.BytesIO(images_bytes))  # Open image first
    img = img.convert("RGB")                  # Then convert

    # set to input size
    img = img.resize((224, 224))

    # convert to array
    img_array = np.array(img)

    # add batch dim
    img_array = np.expand_dims(img_array, axis=0)

    # ResNet50 caffe preprocessing (no TensorFlow needed)
    img_array = _resnet_preprocess(img_array)

    # ensure model is loaded
    await _ensure_model_loaded()

    # run TFLite inference in threadpool to avoid blocking
    def _run_inference(data):
        _interpreter.set_tensor(_input_details[0]['index'], data)
        _interpreter.invoke()
        return _interpreter.get_tensor(_output_details[0]['index'])

    pred_arr = await asyncio.to_thread(_run_inference, img_array)
    pred = float(pred_arr[0][0])

    # decide label and the confidence
    if pred > 0.5:
        prediction = 'Pneumonia'
        confidence = pred * 100
    else:
        prediction = 'Normal'
        confidence = (1 - pred) * 100

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