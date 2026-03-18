import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# We'll expect GROK_API_KEY (Groq) in the environment or .env file
GROK_API_KEY = os.getenv("GROK_API_KEY")

client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

def get_medical_chat_response(message: str, prediction: str, confidence: float, image_b64: str = "", model_id: str = "llama-3.3-70b-versatile"):
    # System prompt to ground the assistant in the model's findings
    system_prompt = f"""You are a helpful and professional Medical Pulmonary Assistant. 
You are given the results of an AI X-ray analysis for a patient.
Current Analysis Results:
- Finding: {prediction}
- Confidence: {confidence:.2f}%

Your goal is to explain these results to the user in a clear, empathetic, and scientifically accurate manner.
Use the provided X-ray image (if available) to describe specifically what you see that correlates with the model's finding.
If the finding is 'Pneumonia', explain what it means, possible symptoms, and recommend seeing a doctor for clinical correlation.
If the finding is 'Normal', reassure the user but remind them that AI is not a substitute for professional medical advice.
Always include a disclaimer that you are an AI assistant and they should consult a medical professional.
Keep your responses concise and informative."""

    # Construct the multimodal message content
    user_content = [{"type": "text", "text": message}]
    
    # Only append image if model is a vision model and image data exists
    if image_b64 and "vision" in model_id.lower():
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        })

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with Groq Vision API: {str(e)}"
