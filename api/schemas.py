from pydantic import BaseModel,Field

class PredcitionResponse(BaseModel):
    prediction : str = Field(description='Prediction made by the Model')
    probability : float
    confidence : float = Field(description='Confidence of the Model')