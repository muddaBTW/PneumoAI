from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from .schemas import PredcitionResponse, ChatRequest, ChatResponse
from .medical_chat import get_medical_chat_response

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
model = tf.keras.models.load_model(str(model_path))

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

    # apply resnet preprocessing
    img_array = preprocess_input(img_array)

    # predictions
    pred = float(model.predict(img_array)[0][0])

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