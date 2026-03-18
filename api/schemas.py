from pydantic import BaseModel,Field

class PredcitionResponse(BaseModel):
    prediction : str = Field(description='Prediction made by the Model')
    probability : float
    confidence : float = Field(description='Confidence of the Model')

class ChatRequest(BaseModel):
    message: str
    prediction_context: str = ""
    confidence_context: float = 0.0
    image_b64: str = "" # Optional base64 image data
    model_id: str = "llama-3.3-70b-versatile" # Default model

class ChatResponse(BaseModel):
    response: str