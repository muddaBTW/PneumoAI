from fastapi import FastAPI,UploadFile,File
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from schemas import PredcitionResponse

app = FastAPI()

# loading out model
model = tf.keras.models.load_model('../model/pneumonia_resnet_model.keras')

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