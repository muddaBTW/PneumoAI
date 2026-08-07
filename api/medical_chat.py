import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# We'll expect GROQ_API_KEY or GROK_API_KEY in the environment or .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

from api.knowledge_base import PNEUMONIA_KNOWLEDGE_BASE

def get_medical_chat_response(message: str, prediction: str, confidence: float, image_b64: str = "", model_id: str = "llama-3.3-70b-versatile"):
    # System prompt to ground the assistant in clinical knowledge and model findings
    system_prompt = f"""You are an expert, empathetic, and professional Medical Pulmonary AI Assistant. 
You are provided with an official Medical Knowledge Base regarding Pneumonia and Chest Radiography:

{PNEUMONIA_KNOWLEDGE_BASE}

Current Patient AI X-Ray Analysis Results:
- Finding: {prediction}
- Confidence: {confidence:.2f}%

Instructions:
1. Ground your answers accurately using the provided Medical Knowledge Base.
2. Explain the AI analysis results in a clear, compassionate, and medically sound manner.
3. If the finding is 'Pneumonia', detail what consolidation/infiltrates mean, mention typical symptoms, highlight red-flag warning signs (e.g. severe dyspnea, cyanosis, high fever), and strongly recommend consulting a physician or pulmonologist for clinical correlation.
4. If the finding is 'Normal', explain what a clear lung fields finding implies while reminding the patient to seek medical evaluation if symptoms persist.
5. Highlight emergency red-flag symptoms when appropriate.
6. ALWAYS include a clear disclaimer that you are an AI assistant and not a substitute for professional medical diagnosis or treatment."""    

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
